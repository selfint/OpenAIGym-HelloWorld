import numpy as np

class Agent:

    def __init__(self, state_space_size, action_space_size, learning_rate, discount_rate, epsilon_decay_rate, min_epsilon_value=0.0):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = 1.0
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon_value = min_epsilon_value
    
    def choose_action(self, state, episode=None):
        """explore epsilon % of the time, exploit if not exploring"""

        # explore -> take a random action
        if np.random.random_sample() < self.epsilon:
            action = np.random.choice(self.action_space_size)

        # exploit -> take the action with the highest expected reward
        else:
            action = np.argmax(self.q_table[state, :])

        # decay epsilon value (set minimum threshold to min epsilon value)
        # exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        if episode is not None:
            self.epsilon = self.min_epsilon_value + (1.0 - self.min_epsilon_value) * np.exp(-self.epsilon_decay_rate * episode)
        return action

    def update_q_table(self, state, action, new_state, reward):
        """update the q table value after observing the reward and the new state following the chosen action"""

        # observe the highest reward possible given the new state
        max_reward_in_new_state = np.max(self.q_table[new_state])

        # observe the previous value of the state action pair
        previous_state_action_value = self.q_table[state, action]

        # update value of the state action pair with respect to learning rate
        self.q_table[state, action] = (1.0 - self.learning_rate) * previous_state_action_value + \
            self.learning_rate * (reward + self.discount_rate * max_reward_in_new_state)

