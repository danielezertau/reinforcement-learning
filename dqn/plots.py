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
    max_reward = 0
    for file, label in zip(files, labels):
        with open(f"{dqn_learn.STATS_DIR}/{file}", 'rb') as f:
            stats = pickle.load(f)
            max_reward = max(max_reward, np.max(stats['best_mean_episode_rewards']))
            plt.plot(stats['best_mean_episode_rewards'][:6000000], label=label)
    plt.axhline(y=max_reward, color='g', linestyle='dashed')
    yticks = list(plt.yticks()[0])
    yticks.remove(20.00)
    plt.yticks(yticks + [max_reward])
    plt.legend()
    plt.savefig(fig_name)
    plt.show()


def annotate_max(stats):
    max_reward = np.max(stats['best_mean_episode_rewards'])
    plt.axhline(y=max_reward, color='g', linestyle='dashed')


if __name__ == '__main__':
    final_p_files = ["statistics-exploration-01.pkl", "statistics-exploration-001.pkl",
                     "statistics-exploration-multiplicative-schedule.pkl", "statistics-orig.pkl"]
    final_p_labels = ["final=0.01", "final=0.001", "multiplicative schedule", "original"]
    q2(files=final_p_files, labels=final_p_labels, fig_name="Q2-final-p.pdf")
    plt.close()
    num_steps_files = ["statistics-exploration-01-25M.pkl", "statistics-exploration-25.pkl",
                       "statistics-exploration-001-25M.pkl", "statistics-orig.pkl"]
    num_steps_labels = ["final=0.01, num_steps=2.5M", "num_steps=2.5M", "final=0.001, num_steps=2.5M", "original"]
    q2(files=num_steps_files, labels=num_steps_labels, fig_name="Q2-num-steps.pdf")
    plt.close()
    q2(files=final_p_files + num_steps_files, labels=final_p_labels + num_steps_labels, fig_name="Q2-all.pdf")
