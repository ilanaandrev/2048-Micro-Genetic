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

#play for one hour
max_total_time = 3600  # 1 hr is 3600 sec
start_time = time.time()

total_max_tile = 0
total_time = 0
highest_tile = 0
games_played = 0

max_tiles = []
times = []

while time.time() - start_time < max_total_time:
    game_start = time.time()

    game = player.play_game(display=None)
    max_tile = game.highest_tile

    game_end = time.time()
    elapsed_game_time = game_end - game_start

    print(f"Game {games_played+1}: Max Tile = {max_tile}, Time = {elapsed_game_time:.4f} seconds")
    
    total_max_tile += max_tile
    total_time += elapsed_game_time
    games_played += 1

    max_tiles.append(max_tile)
    times.append(elapsed_game_time)

    if max_tile > highest_tile:
        highest_tile = max_tile

# After the loop ends (1 hour passed)
print("\n========================")
print(f"Played {games_played} games in 1 hour")
print(f"Average Max Tile: {total_max_tile / games_played:.2f}")
print(f"Average Time per Game: {total_time / games_played:.4f} seconds")
print(f"Highest Tile Achieved: {highest_tile}")
print("========================")