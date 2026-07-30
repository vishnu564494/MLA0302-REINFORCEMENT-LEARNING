import numpy as np

ROWS = 4
COLS = 4

# Rewards
item = (1, 2)
goal = (3, 3)
obstacles = [(1, 1), (2, 2)]

# Discount factor
gamma = 0.9

# Initialize value function
V = np.zeros((ROWS, COLS))

# Actions
actions = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

# Fixed Policy (Always Move Right, else Move Down)
policy = {}

for i in range(ROWS):
    for j in range(COLS):
        if j < COLS - 1:
            policy[(i, j)] = "RIGHT"
        else:
            policy[(i, j)] = "DOWN"

policy[(3, 3)] = None

# Reward Function
def reward(state):
    if state == item:
        return 2
    elif state == goal:
        return 5
    elif state in obstacles:
        return -2
    else:
        return 0

# Next State Function
def next_state(state, action):

    if action is None:
        return state

    x, y = state
    dx, dy = actions[action]

    nx = max(0, min(ROWS - 1, x + dx))
    ny = max(0, min(COLS - 1, y + dy))

    return (nx, ny)

# Policy Evaluation
theta = 0.001

while True:

    delta = 0

    for i in range(ROWS):
        for j in range(COLS):

            state = (i, j)

            action = policy[state]

            ns = next_state(state, action)

            r = reward(ns)

            old_value = V[i, j]

            V[i, j] = r + gamma * V[ns]

            delta = max(delta, abs(old_value - V[i, j]))

    if delta < theta:
        break

# Display Results
print("Policy")
for i in range(ROWS):
    row = []
    for j in range(COLS):
        if policy[(i, j)] is None:
            row.append("GOAL")
        else:
            row.append(policy[(i, j)])
    print(row)

print("\nValue Function")
print(np.round(V, 2))
