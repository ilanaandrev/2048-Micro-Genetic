from abc import ABC, abstractmethod
from game import Action, Game
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class Player(ABC):
    """Abstract player class for 2048. Subclasses will implement `choose_move` for specific behaviour.

    Attributes
    ----------
    scores : List[int]
        The scores for all the games the player has played.
    highest_tiles : List[int]
        The highest tiles for all the games the player has played.
    """

    def __init__(self):
        """Initializes the player with empty scores and highest tiles"""
        self.scores = []
        self.highest_tiles = []

    def get_avg_score(self):
        """Calculate the player's (geometric) average score.

        Returns
        -------
        float
            The geometric mean of self.scores.
        """
        if not self.scores:
            return np.nan
        else:
            return np.exp(np.mean(np.log(self.scores)))

    def get_avg_highest_tile(self):
        """Calculate the player's (geometric) average highest tile.

        Returns
        -------
        float
            The geometric mean of self.highest_tiles.
        """
        if not self.highest_tiles:
            return np.nan
        else:
            return np.exp(np.mean(np.log(self.highest_tiles)))

    def get_num_games_played(self):
        """Get the number of games the player has played.

        Returns
        -------
        int
            The number of games played.
        """
        return len(self.scores)

    def print_summary(self):
        """Print a summary of the player's performance."""
        tiles, counts = np.unique(self.highest_tiles, return_counts=True)
        percents = np.round(100 * counts / np.sum(counts), 1)
        print('Highest Tile Achieved')
        for t, p in zip(tiles, percents):
            print(f'{t}\t{p}%')
        print(f'Average Tile  = {np.rint(self.get_avg_highest_tile()).astype(int)}')
        print(f'Average Score = {np.rint(self.get_avg_score()).astype(int)}')
        print(f'Games Played  = {self.get_num_games_played()}')

    def play_game(self, display):
        """Play a game with optional graphics and add the results to the player's stats.

        Parameters
        ----------
        display : bool
            Whether or not to display graphics
        """
        game = Game()
        if display:
            ax = game.display_board()
        else:
            ax = None
        while not game.game_over:
            action = self._choose_action(game)
            if action == Action.QUIT:
                break
            game.move(action)
            if ax is not None:
                game.display_board(ax)
        if display:
            plt.close()
            print(f'Game Over. Final score was {game.score}. Highest tile was {game.highest_tile}.')
        self.scores.append(game.score)
        self.highest_tiles.append(game.highest_tile)
        return game

    def play_multiple_games(self, num_games, progress_bar=True):
        """Play multiple games without graphics, with an optional tqdm progress bar.

        Parameters
        ----------
        num_games : int
            The number of games to play.
        progress_bar : bool
            Whether or not to display a progress bar.
        """
        if progress_bar:
            iterator = trange(num_games)
        else:
            iterator = range(num_games)
        for _ in iterator:
            self.play_game(False)

    @abstractmethod
    def _choose_action(self, game):
        """Abstract method to determine the next action in the game.

        Parameters
        ----------
        game : Game
            The current game state.

        Returns
        -------
        Action
            The action to take.
        """
        pass
