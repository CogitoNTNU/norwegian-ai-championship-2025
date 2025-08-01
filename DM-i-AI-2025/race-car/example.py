import pygame
import random
from src.game.core import initialize_game_state, game_loop


'''
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
'''

def return_action(state):
    # Returns a list of actions
    actions = []
    action_choices = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
    for _ in range(10):
        actions.append(random.choice(action_choices))
    return actions




if __name__ == '__main__':
    seed_value = 12345
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)
    game_loop(verbose=True) # For pygame window
    pygame.quit()