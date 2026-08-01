import csv
import random

filename = input("Enter CSV file name (Example: grid.csv): ")

grid = []

with open(filename, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        grid.append([cell.strip() for cell in row])

ROWS = len(grid)
COLS = len(grid[0])

print("\nGrid Loaded Successfully:\n")

for row in grid:
    print(" ".join(row))


start = (0, 0)

for i in range(ROWS):
    for j in range(COLS):
        if grid[i][j] == 'S':
            start = (i, j)
            
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

actions = [UP, DOWN, LEFT, RIGHT]

DIRT_REWARD = 1
OBSTACLE_PENALTY = -1
STEP_COST = -0.05

def valid(x, y):
    return 0 <= x < ROWS and 0 <= y < COLS

def move(state, action):
    x, y = state
    dx, dy = action

    nx = x + dx
    ny = y + dy

    if not valid(nx, ny):
        return state, STEP_COST

    if grid[nx][ny] == '#':
        return state, OBSTACLE_PENALTY

    reward = STEP_COST

    if grid[nx][ny] == 'D':
        reward += DIRT_REWARD
        grid[nx][ny] = '.'

    return (nx, ny), reward


def random_policy():
    return random.choice(actions)


def greedy_policy(state):
    x, y = state

    for dx, dy in actions:
        nx = x + dx
        ny = y + dy

        if valid(nx, ny):
            if grid[nx][ny] == 'D':
                return (dx, dy)

    return random.choice(actions)

print("\nSelect Policy")
print("1. Random Policy")
print("2. Greedy Policy")

choice = int(input("Enter your choice: "))

if choice == 1:
    policy = "Random"
else:
    policy = "Greedy"

state = start
total_reward = 0

print("\nRobot Started at:", state)

for step in range(100):

    if policy == "Random":
        action = random_policy()
    else:
        action = greedy_policy(state)

    state, reward = move(state, action)
    total_reward += reward

    print("Step:", step + 1,
          "Position:", state,
          "Reward:", reward)

    dirt_left = False

    for row in grid:
        if 'D' in row:
            dirt_left = True
            break

    if not dirt_left:
        print("\nAll dirt cleaned!")
        break

print("\nFinal Position :", state)
print("Total Reward   :", total_reward)
