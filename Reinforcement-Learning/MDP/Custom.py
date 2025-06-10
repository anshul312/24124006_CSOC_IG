import numpy as np
import itertools
import gymnasium as gym
from gymnasium import spaces

class CustomFrozenLakeEnv(gym.Env):

    def __init__(self, P):
        super().__init__()
        self.P = P
        self.start_state=0
        self.current_state = self.start_state
        self.observation_space = spaces.Discrete(len(P))
        self.action_space = spaces.Discrete(len(P[0]))

    def step(self, action):
        transitions = self.P[self.current_state][action]
        
        # Sample one of the possible transitions based on probability
        probs = [t[0] for t in transitions]
        i = np.random.choice(len(transitions), p=probs)
        prob, next_state, reward, done = transitions[i]
        
        self.current_state = next_state
        return next_state, reward, done, False,{}

    def reset(self,*, seed=None, options=None):
        self.current_state = self.start_state  
        return (self.current_state,{})


#this function takes map as input and outputs the dynamics(env.P) of the environmrnt
def map_to_p(num_row,num_col,ACTIONS,grid):

    num_states = num_row * num_col  

    P = {s: {a: [] for a in ACTIONS} for s in range(num_states)}

    for row, col in itertools.product(range(num_row), range(num_col)):
        s = to_state(row, col,num_col)
        cell_type = grid[row, col]

        for a in ACTIONS:
            li = P[s][a]

            # Terminal state 
            if cell_type in "GH":
                li.append((1.0, s, 0.0, True))
                continue

            # Possible transitions: intended direction or perpendicular to intended
            for b, prob in zip([a, (a - 1)%4, (a + 1)%4], [1/3, 1/3, 1/3]):
                delta = action_to_delta[b]
                new_row = np.clip(row + delta[0], 0, num_row - 1)
                new_col = np.clip(col + delta[1], 0, num_col - 1)
                new_s = to_state(new_row, new_col,num_col)
                new_cell = grid[new_row, new_col]

                # Rewards + done
                reward = 1.0 if new_cell == "G" else 0
                done = new_cell in "GH"

                li.append((prob, new_s, reward, done))

    return P


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTIONS=[LEFT, DOWN, RIGHT, UP]

# Movement vectors
action_to_delta = {
    LEFT:         (0, -1),
    DOWN:         (1, 0),
    RIGHT:        (0, 1),
    UP:           (-1, 0),
}

def to_state(row, col,num_col):
    return row * num_col + col


#First Custom Environment
orig_grid = np.array([list(row) for row in [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]])

custom1_prob = map_to_p(4,4,ACTIONS,orig_grid)

# Second Custom Environment
extended_grid = np.array([list(row) for row in [
    "SFFFFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FFFFFFFH",
    "FFFFFHFF",
    "FFFFFFFH",
    "FFFHFFFF",
    "FFFFFFFG"
]])

custom2_prob = map_to_p(8,8,ACTIONS,extended_grid)

