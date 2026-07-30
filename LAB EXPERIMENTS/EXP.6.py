import numpy as np
import random
import matplotlib.pyplot as plt

ads = 5
rounds = 500
true_ctr = [0.10, 0.15, 0.20, 0.12, 0.18]

def get_click(ad):
    return 1 if random.random() < true_ctr[ad] else 0
  
epsilon = 0.1
counts = np.zeros(ads)
values = np.zeros(ads)
eps_ctr = []

total_clicks = 0

for t in range(rounds):

    if random.random() < epsilon:
        ad = random.randint(0, ads - 1)
    else:
        ad = np.argmax(values)

    reward = get_click(ad)

    counts[ad] += 1
    values[ad] += (reward - values[ad]) / counts[ad]

    total_clicks += reward
    eps_ctr.append(total_clicks)

counts = np.zeros(ads)
values = np.zeros(ads)
ucb_ctr = []

total_clicks = 0

for t in range(rounds):

    if t < ads:
        ad = t
    else:
        ucb = values + np.sqrt((2 * np.log(t + 1)) / counts)
        ad = np.argmax(ucb)

    reward = get_click(ad)

    counts[ad] += 1
    values[ad] += (reward - values[ad]) / counts[ad]

    total_clicks += reward
    ucb_ctr.append(total_clicks)

success = np.ones(ads)
failure = np.ones(ads)

ts_ctr = []

total_clicks = 0

for t in range(rounds):

    sampled = np.random.beta(success, failure)
    ad = np.argmax(sampled)

    reward = get_click(ad)

    if reward == 1:
        success[ad] += 1
    else:
        failure[ad] += 1

    total_clicks += reward
    ts_ctr.append(total_clicks)

print("Total Clicks")
print("--------------------------")
print("Epsilon-Greedy :", eps_ctr[-1])
print("UCB            :", ucb_ctr[-1])
print("Thompson Samp. :", ts_ctr[-1])

plt.figure(figsize=(8,5))

plt.plot(eps_ctr, label="Epsilon-Greedy")
plt.plot(ucb_ctr, label="UCB")
plt.plot(ts_ctr, label="Thompson Sampling")

plt.xlabel("Rounds")
plt.ylabel("Cumulative Clicks")
plt.title("Advertisement Selection using Multi-Armed Bandit")

plt.legend()
plt.grid(True)

plt.show()
