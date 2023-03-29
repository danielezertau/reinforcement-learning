import gymnasium as gym
import numpy
import numpy as np
import matplotlib.pyplot as plt


def agent(w, o):
    return 1 if np.dot(w, o) >= 0 else 0


def run_agent_episode(env, weights):
    observation, info = env.reset()
    total_reward = 0
    truncated = terminated = False
    while not (truncated or terminated):
        action = agent(weights, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    env.close()
    return total_reward


def q3(env):
    weights = np.random.uniform(low=-1, high=1, size=4)
    return run_agent_episode(env, weights)


def random_search(env):
    num_samples = 10000
    weights = np.random.uniform(low=-1, high=1, size=(num_samples, 4))
    vfunc = numpy.vectorize(run_agent_episode, excluded=["env"], signature='(), (n)->()')
    rewards = vfunc(env, weights)

    # Get first index of 200
    return np.argmax(rewards)


def q5(env):
    num_runs = 1000
    results = []
    for i in range(1, num_runs + 1):
        results.append(random_search(env) + 1)
        print(f"Iteration {i} first to 200: {results[-1]}")

    results = np.array(results)
    plt.hist(results)
    plt.axvline(results.mean(), color='k', linestyle='dashed', linewidth=1,
                label='Mean: {:.2f}'.format(results.mean()))
    plt.legend()
    plt.title("Cartpole Episodes Histogram")
    plt.savefig("cartpole.pdf")
    plt.show()


if __name__ == '__main__':
    cartpole = gym.make('CartPole-v0')
    q5(cartpole)
