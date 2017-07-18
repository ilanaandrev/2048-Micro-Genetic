# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:35:57 2017

@author: Costa
"""

import matplotlib.pyplot as plt
import net
import numpy as np
from os import remove
from os.path import isfile


class Population(object):
    """A collection of nets.

    Attributes:
        generation: An integer. The current generation.
        genepool: A list of nets.
    """

    def __init__(self, gen, elite=[], parents=[]):
        """Inits the population.

        Elite are placed directly into the population. Parents compete to have
        kids in the population. If len(elite) < 10 and parents is empty, new
        nets are randomly generated until len(genepool) = 10.

        Args:
            gen: An integer. The generation of new nets.
            elite: A list of nets that are copied directly to the population OR
                   a file name corresponding to a file containing that list.
            parents: A list of nets from which the population will be spawned.
        """
        self.generation = gen

        if type(elite) is str:
            self.genepool = np.load(elite).tolist()
        else:
            self.genepool = elite

        if parents:
            # Get pairs of parents from tournament selection of parents list
            moms, dads = self.tournament(parents)
            for i in range(8):
                self.genepool.append(net.Net(gen, moms[i], dads[i]))
        else:
            current = len(self.genepool)
            for i in range(10 - current):
                self.genepool.append(net.Net(gen))

    def tournament(self, parents):
        """Tournament selection to generate pairs of parents.

        Algorithm works such that the net with the highest score is selected to
        be a parent with probability p, then second highest p*(p-1), third
        highest p*(p-1)**2, etc.

        Args:
            parents: A list of nets.

        Returns:
            moms: A list of 8 nets.
            dads: A list of 8 nets.
        """
        moms = []
        dads = []
        while len(moms) < 8:
            m1, m2, d1, d2 = np.random.choice(parents, 4)
            if m1.score > m2.score:
                moms.append(m1)
            else:
                moms.append(m2)
            if d1.score > d2.score:
                dads.append(d1)
            else:
                dads.append(d2)
        return moms, dads

    def play_games(self, games=50):
        """Get each net in the population to play a certain number of games."""
        for n in self.genepool:
            n.learn_game(games=games)

    def save_generation(self):
        """Write the top 2 nets from every 20th generation to disk.

        Save file from 20 generations ago will be deleted if it exists.

        Precondition: Genepool must be sorted by score.
        """
        if not self.generation % 20 and self.generation != 0:
            name = 'Generation' + str(self.generation)
            np.save(name, self.genepool[:2])
            old_name = 'Generation' + str(self.generation-20) + '.npy'
            if isfile(old_name):
                remove(old_name)

    def sort_by_score(self):
        """Sort the genepool in descending order by each net's score."""
        self.genepool.sort(key=lambda n: n.score, reverse=True)


def train_population(final_gen, initial_gen=0, elite=[]):
    """Run a micro-genetic algorithm to train a (hopefully) good neural net.

    Each population defaluts to 10 nets and plays 50 games. The top 2 from each
    generation are copied to the next one, but all have the opportunity to
    reproduce. Every 10 generations, all but the top 2 are killed and 8 new
    nets are randomly generated to add to the 2 that survived.

    Args:
        final_gen: The total number of generations played is
                   final_gen - initial_gen.  Recommended to be 10*n for
                   positive integer values of n.
        initial_gen: The total number of generations played is
                     final_gen - initial_gen
        elite: List of nets to copy directly into the new population.

    Returns:
        top_scores: List of floats. The top score in each generation.
        best_net: The trained net that performs best.
    """
    parents = []
    top_scores = []

    for gen in range(initial_gen+1, final_gen+1):
        pop = Population(gen, elite, parents)

        print('Playing games for generation', gen, 'of', final_gen)
        pop.play_games()
        pop.sort_by_score()
        pop.save_generation()

        top_scores.append(pop.genepool[0].score)

        elite = pop.genepool[:2]

        if not gen % 10:
            parents = []
        else:
            parents = pop.genepool

        print('Best net\'s generation =', pop.genepool[0].generation)
        print('Best net\'s score =', np.rint((pop.genepool[0].score)), '\n')

    plt.figure()
    plt.title('log_2(Score)')
    plt.plot(np.log2(top_scores))

    print('Finding best...')
    best_net = find_best(pop)

    fname = "BestNetGen" + str(final_gen)
    np.save(fname, best_net)

    print('\a')  # Make noise in Windows
    return top_scores, best_net


def find_best(pop):
    """Play 200 games for each net in population. Print stats and best model.

    Args:
        pop: A population.

    Returns:
        pop.genepool[best]: The best network in pop's genepool.
    """
    pop.play_games(200)
    score = [np.rint(i.score) for i in pop.genepool]
    tile = [np.rint(i.highest_tile) for i in pop.genepool]
    best = np.argmax(score)

    print(score)
    print(tile)
    print('Max score =', np.max(score))
    print('Min score =', np.min(score))
    print('Mean score =', np.mean(score))
    print('Median score =', np.median(score))
    print('Best model =', best)

    return pop.genepool[best]
