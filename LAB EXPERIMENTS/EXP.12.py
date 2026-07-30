import numpy as np
import random

grid = [
    ['S', '.', '.', '#'],
    ['.', '#', '.', '.'],
    ['.', '.', '.', '.'],
    ['#', '.', '.', 'G']
]

ROWS = len(grid)
COLS = len(grid[0])

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

actions = [UP, DOWN, LEFT, RIGHT]

moves = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1)
}

# SARSA Parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

# Q-table
Q = np.zeros((ROWS, COLS, 4))

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)
    return np.argmax(Q[state[0], state[1]])
  
def step(state, action):
    r, c = state
    dr, dc = moves[action]

    nr = r + dr
    nc = c + dc

    # Outside grid
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        return state, -5, False

    # Obstacle
    if grid[nr][nc] == '#':
        return state, -5, False

    # Goal reached
    if grid[nr][nc] == 'G':
        return (nr, nc), 20, True

    # Normal move
    return (nr, nc), -1, False

for episode in range(episodes):

    state = (0, 0)          # Start position
    action = choose_action(state)
    done = False

    while not done:

        next_state, reward, done = step(state, action)

        next_action = choose_action(next_state)

        # -------- SARSA Update --------
        Q[state[0], state[1], action] += alpha * (
            reward +
            gamma * Q[next_state[0], next_state[1], next_action] -
            Q[state[0], state[1], action]
        )

        state = next_state
        action = next_action

symbols = ['↑', '↓', '←', '→']

print("Learned Policy:\n")

for i in range(ROWS):
    for j in range(COLS):

        if grid[i][j] == '#':
            print("#", end=" ")

        elif grid[i][j] == 'G':
            print("G", end=" ")

        else:
            best_action = np.argmax(Q[i, j])
            print(symbols[best_action], end=" ")

    print()
print("\nQ-Table:")
print(np.round(Q, 2))
