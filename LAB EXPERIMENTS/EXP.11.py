import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class StockTradingEnv:

    def __init__(self, prices):
        self.prices = prices
        self.reset()

    def reset(self):
        self.current_step = 0
        self.inventory = []
        self.total_profit = 0
        return np.array([self.prices[self.current_step]])

    def step(self, action):

        reward = 0

        price = self.prices[self.current_step]

        # Buy
        if action == 0:
            self.inventory.append(price)

        # Sell
        elif action == 1 and len(self.inventory) > 0:
            buy_price = self.inventory.pop(0)
            reward = price - buy_price
            self.total_profit += reward

        # Hold -> reward = 0

        self.current_step += 1

        done = self.current_step == len(self.prices) - 1

        next_state = np.array([self.prices[self.current_step]])

        return next_state, reward, done

class DoubleDQN:

    def __init__(self):

        self.state_size = 1
        self.action_size = 3

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=2000)

        self.model = self.build_network()
        self.target_model = self.build_network()

        self.update_target()

    def build_network(self):

        model = Sequential()

        model.add(Dense(24, activation="relu", input_shape=(1,)))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(3, activation="linear"))

        model.compile(loss="mse",
                      optimizer=Adam(learning_rate=0.001))

        return model

    def update_target(self):
        self.target_model.set_weights(
            self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done))

    def act(self, state):

        if random.random() < self.epsilon:
            return random.randint(0,2)

        q = self.model.predict(
            state.reshape(1,1),
            verbose=0)

        return np.argmax(q[0])

    def replay(self, batch_size):

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:

            target = reward

            if not done:

                # Double DQN
                best_action = np.argmax(
                    self.model.predict(
                        next_state.reshape(1,1),
                        verbose=0)[0])

                target = reward + self.gamma * \
                    self.target_model.predict(
                        next_state.reshape(1,1),
                        verbose=0)[0][best_action]

            target_f = self.model.predict(
                state.reshape(1,1),
                verbose=0)

            target_f[0][action] = target

            self.model.fit(
                state.reshape(1,1),
                target_f,
                epochs=1,
                verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

prices = [100,102,101,105,107,110,108,112,
          115,117,116,120,122,121,125]

env = StockTradingEnv(prices)

agent = DoubleDQN()

episodes = 20
batch_size = 8

for episode in range(episodes):

    state = env.reset()

    while True:

        action = agent.act(state)

        next_state, reward, done = env.step(action)

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done)

        state = next_state

        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)

        if done:
            agent.update_target()
            print("Episode:", episode + 1,
                  "Profit:", env.total_profit)
            break

print("\nTraining Completed")
