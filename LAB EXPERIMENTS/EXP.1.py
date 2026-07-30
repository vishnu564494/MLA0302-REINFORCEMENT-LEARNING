import random

GRID_SIZE = 5
dirt = {(0, 2), (2, 2), (4, 4), (3, 1)}
obstacles = {(1, 1), (2, 3), (4, 2)}
actions = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

class GridMDP:
    def __init__(self):
        self.reset()

    def reset(self):
        self.position = (0, 0)
        self.cleaned = set()
        return self.position

    def step(self, action):
        x, y = self.position
        dx, dy = actions[action]

        nx = max(0, min(GRID_SIZE - 1, x + dx))
        ny = max(0, min(GRID_SIZE - 1, y + dy))

        self.position = (nx, ny)

        reward = 0

        if self.position in obstacles:
            reward = -1

        elif self.position in dirt and self.position not in self.cleaned:
            reward = 1
            self.cleaned.add(self.position)

        done = len(self.cleaned) == len(dirt)

        return self.position, reward, done

    def display(self):
        print("\nGrid:")
        for i in range(GRID_SIZE):
            row = ""
            for j in range(GRID_SIZE):
                cell = (i, j)

                if cell == self.position:
                    row += " R "
                elif cell in obstacles:
                    row += " X "
                elif cell in dirt and cell not in self.cleaned:
                    row += " D "
                else:
                    row += " . "
            print(row)
        print()

def random_policy():
    return random.choice(list(actions.keys()))


def greedy_policy(env):
    if len(env.cleaned) == len(dirt):
        return random_policy()

    current = env.position

    remaining = [d for d in dirt if d not in env.cleaned]

    target = min(
        remaining,
        key=lambda x: abs(x[0]-current[0]) + abs(x[1]-current[1])
    )

    tx, ty = target
    x, y = current

    if tx > x:
        return "DOWN"
    elif tx < x:
        return "UP"
    elif ty > y:
        return "RIGHT"
    elif ty < y:
        return "LEFT"

    return random_policy()

def simulate(policy_name, max_steps=50):
    env = GridMDP()
    total_reward = 0

    print("\n==============================")
    print("Policy:", policy_name)
    print("==============================")

    env.display()

    for step in range(max_steps):

        if policy_name == "Random":
            action = random_policy()
        else:
            action = greedy_policy(env)

        state, reward, done = env.step(action)

        total_reward += reward

        print(f"Step {step+1}")
        print("Action:", action)
        print("State :", state)
        print("Reward:", reward)
        print()
        env.display()

        if done:
            print("All dirt cleaned!")
            break

    print("Total Reward =", total_reward)

simulate("Random")
simulate("Greedy")
