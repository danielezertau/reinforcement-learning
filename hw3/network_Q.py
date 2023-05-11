import gym
import numpy as np
import torch
from torch import nn

# Load environment
env = gym.make('FrozenLake-v0')


def one_hot(x):
    one_hot_x = np.zeros(env.observation_space.n, dtype=np.float32)
    one_hot_x[x] = 1
    return torch.from_numpy(one_hot_x)


# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
model = nn.Sequential(nn.Linear(env.observation_space.n, env.action_space.n, bias=False))
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        Q = model(one_hot(s))
        a = torch.argmax(Q).item()

        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            a = env.action_space.sample()

        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(a)

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        Q1 = model(one_hot(s1))

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        Q_target = Q.detach().clone()
        Q_target[a] = r + y * torch.max(Q1).detach().item()

        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        loss = loss_fn(Q_target, Q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1
        if d == True:
            # Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
