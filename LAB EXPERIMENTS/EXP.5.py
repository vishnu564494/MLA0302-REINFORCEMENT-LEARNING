import numpy as np

grid = [
    ['S', 'S', 'S', 'P'],
    ['S', 'X', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'G']
]

ROWS = len(grid)
COLS = len(grid[0])

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

actions = [UP, DOWN, LEFT, RIGHT]
symbols = ['↑', '↓', '←', '→']

gamma = 0.9
theta = 0.001

# Initialize Value Function and Policy
V = np.zeros((ROWS, COLS))
policy = np.zeros((ROWS, COLS), dtype=int)

def next_state(i, j, action):

    ni, nj = i, j

    if action == UP:
        ni -= 1
    elif action == DOWN:
        ni += 1
    elif action == LEFT:
        nj -= 1
    elif action == RIGHT:
        nj += 1

    # Boundary check
    if ni < 0 or ni >= ROWS or nj < 0 or nj >= COLS:
        return i, j

    # Obstacle check
    if grid[ni][nj] == 'X':
        return i, j

    return ni, nj

def reward(i, j):

    if grid[i][j] == 'P':
        return 10
    elif grid[i][j] == 'G':
        return 20
    else:
        return -1


while True:

    delta = 0
    newV = V.copy()

    for i in range(ROWS):
        for j in range(COLS):

            if grid[i][j] == 'X':
                continue

            best_value = -9999
            best_action = 0

            for action in actions:

                ni, nj = next_state(i, j, action)

                value = reward(ni, nj) + gamma * V[ni][nj]

                if value > best_value:
                    best_value = value
                    best_action = action

            newV[i][j] = best_value
            policy[i][j] = best_action

            delta = max(delta, abs(newV[i][j] - V[i][j]))

    V = newV

    if delta < theta:
        break

print("Optimal Dispatch Policy\n")

for i in range(ROWS):
    for j in range(COLS):

        if grid[i][j] == 'X':
            print(" # ", end=" ")

        elif grid[i][j] == 'P':
            print(" P ", end=" ")

        elif grid[i][j] == 'G':
            print(" G ", end=" ")

        else:
            print(symbols[policy[i][j]], end="  ")

    print()
