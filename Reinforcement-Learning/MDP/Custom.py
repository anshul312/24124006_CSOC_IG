import numpy as np
import itertools

class CustomFrozenLakeEnv:
    def __init__(self, P):
        self.P = P
        self.unwrapped = self  # so that env.unwrapped.P works
        self.start_state=0

    def step(self, action):
        transitions = self.P[self.current_state][action]
        
        # Sample one of the possible transitions based on probability
        probs = [t[0] for t in transitions]
        i = np.random.choice(len(transitions), p=probs)
        prob, next_state, reward, done = transitions[i]
        
        self.current_state = next_state
        return next_state, reward, done, {},{}

    def reset(self):
        self.current_state = self.start_state  # typically 0
        return self.current_state


num_row = 4
num_col = 4
num_states = num_row * num_col  
num_actions = 4  

# S=start,F=frozen,H=hole,G=goal
map= np.asarray([
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
], dtype='c')

# Rewards
goal_reward = 1.0
hole_reward = 0.0
default_reward = 0.0

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTIONS=[LEFT, DOWN, RIGHT, UP]

# Movement vectors
action_to_delta = {
    LEFT:  (0, -1),
    DOWN:  (1, 0),
    RIGHT: (0, 1),
    UP:    (-1, 0)
}

def to_state(row, col):
    return row * num_col + col

def to_rc(s):
    return divmod(s, num_col)

P = {s: {a: [] for a in ACTIONS} for s in range(num_states)}

for row, col in itertools.product(range(num_row), range(num_col)):
    s = to_state(row, col)
    cell_type = map[row, col].decode("utf-8")

    for a in ACTIONS:
        li = P[s][a]

        # Terminal state → stay in same state
        if cell_type in "GH":
            li.append((1.0, s, 0.0, True))
            continue

        # Possible transitions: intended, left of intended, right of intended
        for b, prob in zip([a, (a - 1) % 4, (a + 1) % 4], [1/3, 1/3, 1/3]):
            delta = action_to_delta[b]
            new_row = np.clip(row + delta[0], 0, num_row - 1)
            new_col = np.clip(col + delta[1], 0, num_col - 1)
            new_s = to_state(new_row, new_col)
            new_cell = map[new_row, new_col].decode("utf-8")

            # Rewards + done
            reward = goal_reward if new_cell == "G" else (hole_reward if new_cell == "H" else default_reward)
            done = new_cell in "GH"

            li.append((prob, new_s, reward, done))

