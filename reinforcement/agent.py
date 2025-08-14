# reinforcement/agent
import numpy as np
import random
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=1000)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for s, a, r, s2, done in minibatch:
            target = r
            if not done:
                target += self.gamma * np.amax(self.model.predict(s2.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(s.reshape(1, -1), verbose=0)
            target_f[0][a] = target
            self.model.fit(s.reshape(1, -1), target_f, epochs=1, verbose=0)

        if self.epsilon > 0.01:
            self.epsilon *= 0.995