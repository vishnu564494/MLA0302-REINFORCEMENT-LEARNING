import random
grid = [
    ['S', '.', '.', '.'],
    ['.', '#', '.', '.'],
    ['.', '.', 'T', '.'],
    ['.', '.', '.', 'G']
]
ROWS = len(grid)
COLS = len(grid[0])
actions = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

class Road:

    def __init__(self):
        self.position = (0, 0)
        self.goal = (3, 3)

    def move(self, action):

        x, y = self.position
        dx, dy = actions[action]

        nx = x + dx
        ny = y + dy

        if nx < 0 or nx >= ROWS or ny < 0 or ny >= COLS:
            return False

        if grid[nx][ny] == '#':
            return False

        self.position = (nx, ny)
        return True

def safe_policy(env):

    x, y = env.position

    if x < 3 and grid[x+1][y] != '#':
        return "DOWN"

    if y < 3 and grid[x][y+1] != '#':
        return "RIGHT"

    return "UP"

def fast_policy(env):

    x, y = env.position

    if y < 3:
        return "RIGHT"

    if x < 3:
        return "DOWN"

    return "UP"

def random_policy():
    return random.choice(list(actions.keys()))

def simulate(policy_name):

    env = Road()

    steps = 0

    while env.position != env.goal and steps < 30:

        if policy_name == "Safe":
            action = safe_policy(env)

        elif policy_name == "Fast":
            action = fast_policy(env)

        else:
            action = random_policy()

        env.move(action)
        steps += 1

    print(policy_name, "Policy")
    print("Final Position :", env.position)
    print("Steps Taken    :", steps)

    if env.position == env.goal:
        print("Destination Reached")
    else:
        print("Destination Not Reached")

    print()

simulate("Safe")
simulate("Fast")
simulate("Random")
