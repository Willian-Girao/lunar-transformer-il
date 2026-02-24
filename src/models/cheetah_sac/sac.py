"""
https://github.com/DishankJ/Soft-Actor-Critic-PyTorch/tree/main
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, AffineTransform, ComposeTransform, SigmoidTransform
import numpy as np
import os
import gymnasium as gym
from src.models.cheetah_sac.utils import Transition, ReplayBuffer, plot_learning_curve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class Policy(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name, chkpt_dir=os.path.join(project_root, 'saved')):
        super().__init__()
        self.layer1 = nn.Linear(input_dims, 256)
        self.layer2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, n_actions)
        self.logstd = nn.Linear(256, n_actions)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state, get_logprob=False):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        mu = self.mu(x)
        logstd = torch.clamp(self.logstd(x), -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        return action, logprob
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
class Value(nn.Module):
    def __init__(self, lr, input_dims, name, chkpt_dir=os.path.join(project_root, 'saved')):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dims, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
    
    def forward(self, state):
        return self.layers(state)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
class Q(nn.Module):
    def __init__(self, lr, input_dims, n_actions, name, chkpt_dir=os.path.join(project_root, 'saved')):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dims + n_actions, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
        self.to(device)
    
    def forward(self, state, action):
        return self.layers(torch.cat((state, action), dim=1))
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
class SACAgent:
    def __init__(self, input_dims, n_actions, lr=3e-4, max_size=1e6, batch_size=256, gamma=0.99,
                 tau=0.005, reward_scale=2):
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.scale = reward_scale

        torch.manual_seed(123)

        self.policy = Policy(lr, input_dims, n_actions, name='actor')
        self.value = Value(lr, input_dims, name='value')
        self.q1 = Q(lr, input_dims, n_actions, name='q1')
        self.q2 = Q(lr, input_dims, n_actions, name='q2')
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.target_value = Value(lr, input_dims, name='target_value')
        self.replay = ReplayBuffer(int(max_size))

        self.target_value.load_state_dict(self.value.state_dict())
    
    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float, device=device)
        with torch.no_grad():
            action, _ = self.policy(state)
        return action.squeeze().cpu().detach().numpy()
    
    def update_q(self, states, actions, rewards, nextstates, dones):
        q1_loss = F.mse_loss(self.q1(states, actions), self.scale*rewards + (1 - dones)*self.gamma*self.target_value(nextstates))
        q2_loss = F.mse_loss(self.q2(states, actions), self.scale*rewards + (1 - dones)*self.gamma*self.target_value(nextstates))
        return q1_loss, q2_loss

    def update_value(self, states):
        nextactions, logprobs = self.policy(states, get_logprob=True)
        q1_value = self.q1(states, nextactions)
        q2_value = self.q2(states, nextactions)
        min_q_value = torch.min(q1_value, q2_value)
        value_loss = F.mse_loss(self.value(states), min_q_value - logprobs)
        return value_loss
    
    def update_policy(self, states):
        nextactions, logprobs = self.policy(states, get_logprob=True)
        q1_value = self.q1(states, nextactions)
        q2_value = self.q2(states, nextactions)
        min_q_value = torch.min(q1_value, q2_value)
        policy_loss = (logprobs - min_q_value).mean()
        return policy_loss

    def learn(self):
        if self.replay.memory.__len__() < self.batch_size:
            return
        
        for i in range(1):
            samples = self.replay.sample(self.batch_size)

            states = torch.tensor(samples.state, device=device, dtype=torch.float)
            actions = torch.tensor(samples.action, device=device, dtype=torch.float)
            nextstates = torch.tensor(samples.nextstate, device=device, dtype=torch.float)
            rewards = torch.tensor(samples.reward, device=device, dtype=torch.float).view(-1, 1)
            dones = torch.tensor(samples.done, device=device, dtype=torch.float).view(-1, 1)

            self.value.optimizer.zero_grad()
            value_loss = self.update_value(states)
            value_loss.backward()
            self.value.optimizer.step()

            self.q_optimizer.zero_grad()
            q1_loss, q2_loss = self.update_q(states, actions, rewards, nextstates, dones)
            q_loss = q1_loss + q2_loss
            q_loss.backward()
            self.q_optimizer.step()

            self.policy.optimizer.zero_grad()
            policy_loss = self.update_policy(states)
            policy_loss.backward()
            self.policy.optimizer.step()

            with torch.no_grad():
                for target_value_param, value_param in zip(self.target_value.parameters(), self.value.parameters()):
                    target_value_param.data.copy_(self.tau * value_param.data + (1.0 - self.tau) * target_value_param.data)

    def save_models(self):
        self.policy.save_checkpoint()
        self.q1.save_checkpoint()
        self.q2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        self.policy.load_checkpoint()
        self.q1.load_checkpoint()
        self.q2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v5')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim, reward_scale=2, tau=0.005)

    total_timesteps = 1e6
    saving_interval = 20000    
    n_random_actions = 5000

    episode_rewards = []
    timestep = 0
    while timestep < total_timesteps:
        done = False
        obs, _ = env.reset()
        episode_reward = 0
        while not done:
            if timestep < n_random_actions:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(obs.tolist())
            obs_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay.push(Transition(obs.tolist(), action.tolist(), reward, obs_.tolist(), done))
            obs = obs_
            agent.learn()
            episode_reward += reward
            timestep += 1
        episode_rewards.append(episode_reward)
        avg_score = np.mean(episode_rewards[-100:])
        print(f'{timestep=}, episode_reward={episode_reward:.2f}, avg_score={avg_score:.2f}')

        if timestep % saving_interval == 0:
            agent.save_models()

    plot_learning_curve(episode_rewards)