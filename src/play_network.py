import os
import pickle
import time
from players.network import NetworkPlayer

# using the pre-trained model
this_dir = os.path.dirname(os.path.abspath(__file__))
path_to_model = os.path.join(this_dir, '..', 'Best_Net_Gen_2000.pkl')

with open(path_to_model, 'rb') as f:
    network = pickle.load(f)

player = NetworkPlayer(network)

num_games = 1000
total_score = 0
total_max_tile = 0
total_time = 0
highest_tile = 0

scores = []
max_tiles = []
times = []

for i in range(num_games):
    start_time = time.time()  # Start timing

    game = player.play_game(display=None)
    score = game.score
    max_tile = game.highest_tile

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time

    print(f"Game {i+1}: Score = {score}, Max Tile = {max_tile}, Time = {elapsed_time:.4f} seconds")

    total_score += score
    total_max_tile += max_tile
    total_time += elapsed_time

    scores.append(score)
    max_tiles.append(max_tile)
    times.append(elapsed_time)

    if max_tile > highest_tile:  # <- update highest tile
        highest_tile = max_tile

# Print average results
print("\n========================")
print(f"Played {num_games} games")
print(f"Average Score: {total_score / num_games:.2f}")
print(f"Average Max Tile: {total_max_tile / num_games:.2f}")
print(f"Average Time per Game: {total_time / num_games:.4f} seconds")
print(f"Highest Tile Achieved: {highest_tile}")
print("========================")