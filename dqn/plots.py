import pickle
import matplotlib.pyplot as plt
import numpy as np
import dqn_learn


def q1():
    with open(dqn_learn.STATS_FILE, 'rb') as f:
        stats = pickle.load(f)
    plt.title("DQN Basic Results")
    plt.xlabel('Num Timesteps')
    plt.ylabel('Rewards')
    plt.plot(stats['mean_episode_rewards'][:6000000], label="Mean 100-Episode Reward")
    plt.plot(stats['best_mean_episode_rewards'][:6000000], label="Best Mean Reward")
    max_reward = np.max(stats['best_mean_episode_rewards'])
    plt.axhline(y=max_reward, color='g', linestyle='dashed')
    plt.legend()
    plt.savefig("Q1.pdf")
    plt.show()


def q2(files, labels, fig_name):
    plt.title("DQN Hyper-Parameters Tuning")
    plt.xlabel('Num Timesteps')
    plt.ylabel('Rewards')
    for file, label in zip(files, labels):
        with open(f"{dqn_learn.STATS_DIR}/{file}", 'rb') as f:
            stats = pickle.load(f)
            plt.plot(stats['best_mean_episode_rewards'][:6000000], label=label)
    plt.legend()
    plt.savefig(fig_name)
    plt.show()


if __name__ == '__main__':
    q2(files=["statistics-exploration-01.pkl", "statistics-exploration-001.pkl", "statistics-orig.pkl"],
       labels=["final=0.01", "final=0.001", "original"], fig_name="Q2-final-p.pdf")
    q2(files=["statistics-exploration-01-25M.pkl", "statistics-exploration-25.pkl",
              "statistics-exploration-001-25M.pkl", "statistics-orig.pkl"],
       labels=["final=0.01, num_steps=2.5M", "num_steps=2.5M", "final=0.001, num_steps=2.5M", "original"],
       fig_name="Q2-num-steps.pdf")
