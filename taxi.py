# Q-Learning on Taxi-v3 (OpenAI Gym)
# ----------------------------------
# Prerequisite: `pip install gym`
#
# This script trains a Q-learning agent to solve the Taxi-v3 environment.
# It handles Gym v0.26+ step API differences and demonstrates training,
# evaluation, and optional rendering.

import gym
import numpy as np

def train_q_learning(env_name="Taxi-v3",
                     alpha=0.1, gamma=0.99,
                     epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                     n_episodes=5000, max_steps=100):
    """
    Train a Q-learning agent on the specified Gym environment.
    Returns the learned Q-table and the list of rewards per episode.
    """
    # Create environment (no rendering during training)
    env = gym.make(env_name)

    n_states = env.observation_space.n   # Number of discrete states
    n_actions = env.action_space.n       # Number of discrete actions

    # Initialize Q-table: shape (states, actions)
    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        # For Gym >=0.26 .reset() returns (obs, info)
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0

        for step in range(max_steps):
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            # Take action and unpack result
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            # Q-learning update (Bellman equation)
            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += reward
            state = next_state
            if done:
                break

        # Decay exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    env.close()
    return Q, rewards_per_episode

def evaluate_policy(Q, env_name="Taxi-v3", n_episodes=100):
    """
    Evaluate the learned Q-table policy over a number of episodes.
    Returns the list of rewards collected.
    """
    env = gym.make(env_name)
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[state])  # Always exploit
            result = env.step(action)
            if len(result) == 5:
                state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                state, reward, done, _ = result
            total_reward += reward
        rewards.append(total_reward)
    env.close()
    return rewards

if __name__ == "__main__":
    # Hyperparameters
    Q_table, training_rewards = train_q_learning()

    # Evaluate the learned policy
    eval_rewards = evaluate_policy(Q_table)

    print(f"Average training reward over {len(training_rewards)} episodes: "
          f"{np.mean(training_rewards):.2f}")
    print(f"Average evaluation reward over {len(eval_rewards)} episodes: "
          f"{np.mean(eval_rewards):.2f}")

