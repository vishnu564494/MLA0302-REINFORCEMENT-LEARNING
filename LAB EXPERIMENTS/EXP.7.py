import numpy as np
import matplotlib.pyplot as plt
grid = [
    ['S', 'S', 'S', 'S'],
    ['S', '#', 'S', 'S'],
    ['S', 'S', 'S', 'G'],
    ['S', 'S', 'S', 'S']
]

ROWS = len(grid)
COLS = len(grid[0])

gamma = 0.9
theta = 0.001

# Actions
RIGHT = 0
DOWN = 1

def reward(i, j):
    if grid[i][j] == 'G':
        return 10
    elif grid[i][j] == '#':
        return -5
    else:
        return -1


def next_state(i, j, action):

    ni, nj = i, j

    if action == RIGHT:
        nj += 1
    elif action == DOWN:
        ni += 1

    # Boundary check
    if ni < 0 or ni >= ROWS or nj < 0 or nj >= COLS:
        return i, j

    # Obstacle
    if grid[ni][nj] == '#':
        return i, j

    return ni, nj

def value_function(policy):

    V = np.zeros((ROWS, COLS))

    while True:

        delta = 0

        newV = V.copy()

        for i in range(ROWS):
            for j in range(COLS):

                if grid[i][j] == '#':
                    continue

                ni, nj = next_state(i, j, policy)

                newV[i][j] = reward(ni, nj) + gamma * V[ni][nj]

                delta = max(delta, abs(newV[i][j] - V[i][j]))

        V = newV

        if delta < theta:
            break

    return V


right_policy = value_function(RIGHT)
down_policy = value_function(DOWN)

print("Value Function - Right Policy")
print(np.round(right_policy,2))

print("\nValue Function - Down Policy")
print(np.round(down_policy,2))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(right_policy, cmap='viridis')
plt.title("Right Policy")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(down_policy, cmap='viridis')
plt.title("Down Policy")
plt.colorbar()

plt.show()
