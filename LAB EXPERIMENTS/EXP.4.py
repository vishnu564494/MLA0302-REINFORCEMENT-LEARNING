import numpy as np
grid = [
    ['S', '.', '.', '.'],
    ['.', '#', '.', '.'],
    ['.', '.', '.', '.'],
    ['#', '.', '.', 'G']
]

ROWS = len(grid)
COLS = len(grid[0])

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

actions = [UP, DOWN, LEFT, RIGHT]
gamma = 0.9
policy = np.full((ROWS, COLS), RIGHT)

V = np.zeros((ROWS, COLS))

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

    # Boundary Check
    if ni < 0 or ni >= ROWS or nj < 0 or nj >= COLS:
        return i, j

    # Obstacle Check
    if grid[ni][nj] == '#':
        return i, j

    return ni, nj

def reward(i, j):

    if grid[i][j] == 'G':
        return 10

    return -1


stable = False

while not stable:

    while True:

        delta = 0

        for i in range(ROWS):
            for j in range(COLS):

                if grid[i][j] == '#':
                    continue

                ni, nj = next_state(i, j, policy[i][j])

                value = reward(ni, nj) + gamma * V[ni][nj]

                delta = max(delta, abs(value - V[i][j]))

                V[i][j] = value

        if delta < 0.001:
            break

    stable = True

    for i in range(ROWS):
        for j in range(COLS):

            if grid[i][j] == '#':
                continue

            old_action = policy[i][j]

            best_action = old_action
            best_value = -9999

            for action in actions:

                ni, nj = next_state(i, j, action)

                value = reward(ni, nj) + gamma * V[ni][nj]

                if value > best_value:
                    best_value = value
                    best_action = action

            policy[i][j] = best_action

            if old_action != best_action:
                stable = False

symbols = ['↑', '↓', '←', '→']

print("Optimal Policy\n")

for i in range(ROWS):

    for j in range(COLS):

        if grid[i][j] == '#':
            print(" # ", end=" ")

        elif grid[i][j] == 'G':
            print(" G ", end=" ")

        else:
            print(symbols[policy[i][j]], end="  ")

    print()

print("\nValue Function\n")
print(np.round(V, 2))
