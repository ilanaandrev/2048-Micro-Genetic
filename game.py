# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:47:42 2017

@author: Costa
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import win32api as wapi


class Game(object):
    """A single game of 2048.

    A game consists of a 4x4 board of tiles with integer values representing
    the log2 of the tile's value.  The tiles can be shifted up, down, left, or
    right, and when two tiles of the same value touch, they merge and their
    value is doubled.  If the board is full and there are no legal moves, the
    game ends.

    Attributes:
        board: A 4x4 numpy array of the log2 tile value at each board position.
        score: An integer. The current score.
        highest_tile: An integer. The highest-value tile on the board.
        moves: An integer. The number of moves made in the current game.
        game_over: A boolean describing whether or not the game is over.
        last_move_illegal: A boolean to flag illegal moves.
    """

    def __init__(self):
        """Inits Game class.

        Places two tiles randomly on the board and sets highest_tile
        accordingly. Score starts at 0, game_over and last_move_illegal are
        False, and clears the key-state.
        """
        self.board = np.zeros((4, 4), dtype=np.int)
        self.add_tile()
        self.add_tile()
        self.score = 0
        self.highest_tile = 0
        self.moves = 0
        self.game_over = False
        self.last_move_illegal = False
        for key in [37, 38, 39, 40, 81]:
            # In order, [left, up, right, down, Q]
            wapi.GetAsyncKeyState(key)

    def add_tile(self):
        """Adds a 2 or 4 tile randomly to the current game board.

        Precondition: Board must not already be full.
        """
        # 10% chance the new tile is a 4, else it's a 2.
        if np.random.random() > 0.9:
            val = 2
        else:
            val = 1
        # Choose a random empty space to add the tile to
        board = self.board.reshape(16)
        valid_pos = [i for i in range(16) if not board[i]]
        pos = np.random.choice(valid_pos)
        board[pos] = val
        self.board = board.reshape((4, 4))
        # Game over contition: Board is full and no legal moves.
        if self.board.all() and self.no_moves():
            self.game_over = True

    def no_moves(self):
        """Return True if there are no legal moves, else false."""
        orig_board = np.copy(self.board)
        orig_score = np.copy(self.score).tolist()
        # Moves in order [left, up, right, down]
        for key in [37, 38, 39, 40]:
            self.move(key)
            self.board = orig_board
            self.score = orig_score
            if not self.last_move_illegal:
                return False
            else:
                self.last_move_illegal = False
        return True

    def read_key(self):
        """Read key from keyboard.

        Returns:
            key: An integer representing the ASCII value of the key pressed.
        """
        while True:
            time.sleep(0.1)
            for key in [37, 38, 39, 40, 81]:
                # In order, [left, up, right, down, Q]
                if wapi.GetAsyncKeyState(key):
                    return key

    def slide_left(self):
        """Slide tiles left one column at a time, but don't merge them."""
        for row in range(4):
            new_row = [i for i in self.board[row, :] if i != 0]
            new_row = new_row + [0 for i in range(4-len(new_row))]
            self.board[row, :] = np.array(new_row)

    def merge_left(self):
        """Merge tiles and increase score according to 2048 rules."""
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j+1] != 0:
                    self.board[i, j] = self.board[i, j] + 1
                    self.board[i, j+1] = 0
                    self.score += 2**self.board[i, j]

    def move(self, key):
        """Execute move in direction of key.

        Rotate the board so that key points leftwards, move left, then rotate
        back to the original position.

        Args:
            key: An integer representing the ASCII value of the key pressed.
        """
        prev_board = np.copy(self.board)
        if key == 37:
            rot = 0
        elif key == 38:
            rot = 1
        elif key == 39:
            rot = 2
        else:
            rot = -1
        self.board = np.rot90(self.board, rot)
        # Slide, merge, slide pattern reflects official 2048 rules
        self.slide_left()
        self.merge_left()
        self.slide_left()
        self.board = np.rot90(self.board, -1*rot)
        # If the board is unchanged by the move, then it was illegal.
        if np.array_equal(self.board, prev_board):
            self.last_move_illegal = True

    def process_key(self, key):
        """Quit, or move according to key. Ignore illegal moves.

        Args:
            key: An integer representing the ASCII value of the key pressed.
        """
        # ord('Q') = 81.  Quit game.
        if key == 81:
            self.game_over = True
        else:
            self.move(key)
            if not self.last_move_illegal:
                self.add_tile()
                self.moves += 1
        self.highest_tile = np.max(2**self.board)

    def display_game(self, axes=None):
        """Display board using pyplot.

        Args:
            axes: Pyplot axes. Use to plot over previous figure.

        Returns:
            axes: Pyplot axes for the current figure.
        """
        if not axes:
            fig, axes = plt.subplots()
            show_flag = True  # Whether or not we need to run plt.show()
            plt.pause(0.5)  # Gives you a bit of time to adjust the plot
        else:
            show_flag = False
            axes.clear()
        axes.set_title('Score = ' + str(self.score))
        plt.imshow(self.board, cmap='summer')
        # Print numbers on tile if log2(tile) != 1
        for (j, i), label in np.ndenumerate(2**self.board):
            if label != 1:
                axes.text(i, j, label, ha='center', va='center')
        plt.pause(0.001)  # Need a slight delay or the plot won't render
        if show_flag:
            plt.show()
        return axes

    def play_game(self):
        """Play a game of 2048."""
        ax = self.display_game()
        while not self.game_over:
            key = self.read_key()
            self.last_move_illegal = False
            self.process_key(key)
            self.display_game(ax)
        print('Game Over')
