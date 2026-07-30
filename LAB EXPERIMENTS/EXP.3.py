import numpy as np
import random
import matplotlib.pyplot as plt

prices = [10, 20, 30, 40, 50]
true_rewards = [15, 25, 40, 35, 20]

num_arms = len(prices)
num_rounds = 500

def get_reward(arm):
    return np.random.normal(true_rewards[arm], 3)


epsilon = 0.1

counts = np.zeros(num_arms)
values = np.zeros(num_arms)

eps_total = []

reward_sum = 0

for t in range(num_rounds):

    if random.random() < epsilon:
        arm = random.randint(0, num_arms - 1)
    else:
        arm = np.argmax(values)

    reward = get_reward(arm)

    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]

    reward_sum += reward
    eps_total.append(reward_sum)


counts = np.zeros(num_arms)
values = np.zeros(num_arms)

ucb_total = []

reward_sum = 0

for t in range(num_rounds):

    if t < num_arms:
        arm = t
    else:
        ucb = values + np.sqrt((2 * np.log(t + 1)) / counts)
        arm = np.argmax(ucb)

    reward = get_reward(arm)

    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]

    reward_sum += reward
    ucb_total.append(reward_sum)


success = np.ones(num_arms)
failure = np.ones(num_arms)

ts_total = []

reward_sum = 0

threshold = 30

for t in range(num_rounds):

    samples = np.random.beta(success, failure)
    arm = np.argmax(samples)

    reward = get_reward(arm)

    if reward >= threshold:
        success[arm] += 1
    else:
        failure[arm] += 1

    reward_sum += reward
    ts_total.append(reward_sum)

print("Final Revenue")
print("-------------------------")
print("Epsilon-Greedy :", round(eps_total[-1], 2))
print("UCB            :", round(ucb_total[-1], 2))
print("Thompson Samp. :", round(ts_total[-1], 2))

plt.figure(figsize=(8,5))

plt.plot(eps_total, label="Epsilon-Greedy")
plt.plot(ucb_total, label="UCB")
plt.plot(ts_total, label="Thompson Sampling")

plt.xlabel("Pricing Decisions")
plt.ylabel("Cumulative Revenue")
plt.title("Dynamic Pricing using Multi-Armed Bandit")

plt.legend()
plt.grid(True)

plt.show()
