import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


#Policy Network
class ActorCNN(nn.Module):
    def __init__(self, action_dim, max_action):
        super(ActorCNN, self).__init__()

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.lin2 = nn.Linear(512, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        # this is the vanilla implementation
        # but we're using a slightly different one
        # x = self.max_action * self.tanh(self.lin2(x))

        # because we don't want our duckie to go backwards

        x_mean = self.lin2(x)
        x_std = self.lin2(x)

        x_std = torch.clamp(x_std, min = -0.2, max = 0.2)

        return x_mean, x_std


    def evaluate(self, state, epsilon = 1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = mean + std * z.to(device)
        action[:, 0] = self.max_action * self.sigm(action[:, 0])
        action[:, 1] = self.tanh(action[:, 1])

        log_prob = torch.sum(Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon))

        return action, log_prob


#Value Network
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, states):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(x)) 
        x = self.lin3(x)

        return x




#Soft Q Network
class CriticCNN(nn.Module):
    def __init__(self, action_dim):
        super(CriticCNN, self).__init__()

        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, states, actions):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(torch.cat([x, actions], 1)))  # c
        x = self.lin3(x)

        return x


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, net_type):
        super(SAC, self).__init__()
        print("Starting SAC init")
        assert net_type in ["cnn"]

        self.state_dim = state_dim
        self.flat = False

        
        self.value_net = ValueNetwork().to(device)
        self.target_value_net = ValueNetwork().to(device)
        print("Initialized Value Net")


        self.soft_q_net1 = CriticCNN(action_dim).to(device)
        self.soft_q_net2 = CriticCNN(action_dim).to(device)
        print("Initialized Soft Q")

        self.policy_net = ActorCNN(action_dim, max_action).to(device)
        self.target_policy_net = ActorCNN(action_dim, max_action).to(device)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        print("Initialized Policy Net")

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-4)
        self.soft_q_net1_optimizer = torch.optim.Adam(self.soft_q_net1.parameters(), lr=1e-4)
        self.soft_q_net2_optimizer = torch.optim.Adam(self.soft_q_net2.parameters(), lr=1e-4)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        print("Initialized Optimizers")


        self.value_criterion = torch.nn.MSELoss()
        self.soft_q_criterion1 = torch.nn.MSELoss()
        self.soft_q_criterion2 = torch.nn.MSELoss()
        
    def predict(self, state):

        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3
        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.policy_net.evaluate(state)[0].cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        for it in range(iterations):

            # Sample replay buffer, Randomly sample a batch of transitions
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            # Compute predicted values for Q and V functions
            predicted_q_value1 = self.soft_q_net1(state, action)
            predicted_q_value2 = self.soft_q_net2(state, action)

            predicted_value = self.value_net(state)

            new_action, log_prob = self.policy_net.evaluate(state)

            # Compute target Q values and Update Q-functions by one step of gredient descent
            target_value = self.target_value_net(next_state)
            target_q_value = reward + done * discount * target_value

            q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
            q_value_loss2 = self.soft_q_criterion1(predicted_q_value2, target_q_value.detach())

            self.soft_q_net1_optimizer.zero_grad()
            q_value_loss1.backward()
            self.soft_q_net1_optimizer.step()

            self.soft_q_net2_optimizer.zero_grad()
            q_value_loss2.backward()
            self.soft_q_net2_optimizer.step()


            # Update V-functions by one step of gredient descent
            predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
            target_value_func = predicted_new_q_value - log_prob
            value_loss = self.value_criterion(predicted_value, target_value_func.detach())

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Update policy by one step of gradient ascent
            policy_loss = (log_prob - predicted_new_q_value).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update target value network
            for param, target_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)




    def save(self, filename, directory):
        print("Saving to {}/{}_[actor|critic].pth".format(directory, filename))
        torch.save(self.policy_net.state_dict(), '{}/{}_actor.pth'.format(directory, filename))
        print("Saved Actor")
        #torch.save(self.critic.state_dict(), '{}/{}_critic.pth'.format(directory, filename))
        #print("Saved Critic")
        
    def load(self, filename, directory):
        self.policy_net.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, filename), map_location=device))
        #self.critic.load_state_dict(torch.load('{}/{}_critic.pth'.format(directory, filename), map_location=device))
