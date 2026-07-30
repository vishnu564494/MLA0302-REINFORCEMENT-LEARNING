import numpy as np
actions = ["Buy", "Hold", "Sell"]

# Rewards for each action
rewards = [5, 2, -2]

# Initial Policy Probabilities
policy = np.array([0.33, 0.34, 0.33])

learning_rate = 0.05
episodes = 100

print("Initial Policy:", np.round(policy, 2))

for episode in range(episodes):

    # Choose action according to policy
    action = np.random.choice(3, p=policy)

    reward = rewards[action]

    # Policy Gradient Update
    policy[action] += learning_rate * reward

    # Keep probabilities positive
    policy = np.clip(policy, 0.01, None)

    # Normalize probabilities
    policy = policy / np.sum(policy)

print("\nFinal Policy Probabilities")

for i in range(3):
    print(actions[i], ":", round(policy[i], 3))

best_action = actions[np.argmax(policy)]

print("\nBest Investment Strategy:", best_action)
