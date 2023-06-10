import pickle
import matplotlib.pyplot as plt
import numpy as np

import dqn_learn

if __name__ == '__main__':
    with open(dqn_learn.STATS_FILE, 'rb') as f:
        stats = pickle.load(f)
    plt.title("DQN Basic Results")
    plt.xlabel('Num Timesteps')
    plt.ylabel('Rewards')
    plt.plot(stats['mean_episode_rewards'][:6000000], label="Mean 100-Episode Reward")
    plt.plot(stats['best_mean_episode_rewards'][:6000000], label="Best Mean Reward")
    max_reward = np.max(stats['best_mean_episode_rewards'])
    plt.axhline(y=max_reward, color='g', linestyle='dashed')
    yticks = list(plt.yticks()[0])
    yticks.remove(20.00)
    plt.yticks(yticks + [max_reward])
    plt.legend()
    plt.savefig("Q1.pdf")
    plt.show()
