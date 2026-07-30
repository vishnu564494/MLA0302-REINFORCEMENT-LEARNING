import random
import numpy as np
states = ["Normal", "Priority", "VIP"]

# Rewards
rewards = {
    "Normal": 2,
    "Priority": 5,
    "VIP": 8
}

# Number of Episodes
episodes = 1000

def random_policy():
    return random.choice(states)


def priority_policy():

    r = random.random()

    if r < 0.6:
        return "Priority"
    elif r < 0.9:
        return "VIP"
    else:
        return "Normal"

def monte_carlo(policy):

    returns = {s: [] for s in states}

    for episode in range(episodes):

        state = policy()

        reward = rewards[state]

        returns[state].append(reward)

    value = {}

    for s in states:
        value[s] = np.mean(returns[s])

    return value

random_values = monte_carlo(random_policy)
priority_values = monte_carlo(priority_policy)


print("Estimated Value Function\n")

print("Random Policy")
for s in states:
    print(s, ":", round(random_values[s],2))

print("\nPriority Policy")
for s in states:
    print(s, ":", round(priority_values[s],2))
