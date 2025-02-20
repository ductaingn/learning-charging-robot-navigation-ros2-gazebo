from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class FDM(nn.Module):
    def __init__(self, history_dim, obs_dim, output_dim, *args, **kwargs):
        super(FDM, self).__init__(*args, **kwargs)

        self.history_embedding = nn.Linear(history_dim, 128)
        self.current_embedding = nn.Linear(obs_dim, 128)

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

        self.mha = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)

        self.layer_norm = nn.LayerNorm(128)

        self.encode = nn.Linear(128, output_dim)


    def forward(self, history_observation, current_observation):
        '''
        history_observation: shape = (batch_size, n_history_frame, lidar_dim)
        current_observation: shape = (batch_size, lidar_dim)

        out: shape = (batch_size, output_dim)
        '''
        history_observation = self.history_embedding(history_observation)
        
        hs, self.hidden_his_state = self.lstm(history_observation)
        hs = F.leaky_relu(hs)
            
        co = F.leaky_relu(self.current_embedding(current_observation))

        co = F.leaky_relu(self.fc1(co))
        co = F.leaky_relu(self.fc2(co))

        concat = torch.cat([hs, co], dim=1)
        
        out, _ = self.mha(concat, concat, concat)

        out = self.encode(out)

        return out

class KineticModel(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super(KineticModel, self).__init__(*args, **kwargs)

        self.kinetic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim//2)
        )

        self.waypoint_encoder = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first=True, proj_size=output_dim//4, dropout=0.1)

        self.goal_encoder = nn.Linear(2, output_dim//4)

    def forward(self, kinetic_state, waypoints, goal):
        n_envs = kinetic_state.shape[0]
        kinetic_state = self.kinetic(kinetic_state)
        
        out, (hn, cn) = self.waypoint_encoder(waypoints)
        waypoints = out[:,-1,:].unsqueeze(1)
        
        goal = self.goal_encoder(goal)

        return torch.cat([kinetic_state, waypoints, goal], dim=-1)
    

class BackBone(nn.Module):
    def __init__(self, history_dim, observation_dim, kin_input_dim, kin_output_dim, hidden_dim, n_history_frame, *args, **kwargs):
        super(BackBone, self).__init__(*args, **kwargs)
        self.observation_dim = observation_dim
        self.kin_input_dim = kin_input_dim
        self.n_history_frame = n_history_frame
        self.hidden_dim = hidden_dim
        self.history_dim = history_dim

        self.fdm = FDM(history_dim, observation_dim, kin_output_dim)
        self.kinetic_model = KineticModel(kin_input_dim, kin_output_dim)

        self.mha = nn.MultiheadAttention(kin_output_dim, 4, dropout=0.1, batch_first=True)

        self.encode = nn.Sequential(
            nn.Linear(kin_output_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, hidden_dim),
            nn.LeakyReLU()
        )

    def forward(self, state):
        '''
        state: shape = (batch_size, kin_input_dim + observation_dim*(history_dim+1) + 2 + 2*n_waypoints)

        latent: shape = (batch_size, hidden_dim)
        '''
        kinetic_state = state[:, :self.kin_input_dim]
        current_observation = state[:, self.kin_input_dim:self.kin_input_dim + self.observation_dim]
        history_observation = state[:, self.kin_input_dim + self.observation_dim:self.kin_input_dim + self.observation_dim + self.history_dim * self.n_history_frame]
        goals = state[:, self.kin_input_dim + self.observation_dim + self.history_dim * self.n_history_frame:self.kin_input_dim + self.observation_dim + self.history_dim * self.n_history_frame + 2]
        waypoints = state[:, self.kin_input_dim + self.observation_dim + self.history_dim * self.n_history_frame + 2:]

        kinetic_state = kinetic_state.reshape(-1, 1, self.kin_input_dim)
        history_observation = history_observation.reshape(-1, self.n_history_frame, self.observation_dim)
        current_observation = current_observation.reshape(-1, 1, self.observation_dim)
        n_envs = kinetic_state.shape[0]
        waypoints = waypoints.reshape(n_envs, -1, 2)
        goals = goals.reshape(n_envs, 1, 2)

        env_rep = self.fdm.forward(history_observation, current_observation)
        robot_rep = self.kinetic_model.forward(kinetic_state, waypoints, goals)

        concat = torch.cat([env_rep, robot_rep], dim=1)

        out, _ = self.mha(concat, concat, concat)

        latent = self.encode(out)
        latent = torch.sum(latent, dim=1).reshape(-1, self.hidden_dim)

        return latent
    

class DeterministicActor(nn.Module):
    def __init__(self, history_dim, observation_dim, kin_input_dim, kin_output_dim, command_dim, n_history_frame=16, action_space=None, *args, **kwargs):
        super(DeterministicActor, self).__init__(*args, **kwargs)

        self.backbone = BackBone(history_dim, observation_dim, kin_input_dim, kin_output_dim, 128, n_history_frame)

        self.mean = nn.Linear(128, command_dim)
        self.noise = torch.Tensor(command_dim)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(5.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)


    def forward(self, state):
        '''
        state: shape = (batch_size, observation_dim=361)

        command: shape = (batch_size, command_dim)
        '''
        latent = self.backbone(state)

        command = F.tanh(self.mean(latent))*self.action_scale + self.action_bias

        return command
    
    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean
    
    def to(self, device):
        self.backbone.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicActor, self).to(device)
    

class GaussianActor(nn.Module):
    def __init__(self, history_dim, observation_dim, kin_input_dim, kin_output_dim, command_dim, n_history_frame=16, action_space=None, *args, **kwargs):
        super(GaussianActor, self).__init__(*args, **kwargs)

        self.backbone = BackBone(history_dim, observation_dim, kin_input_dim, kin_output_dim, 128, n_history_frame)

        self.mean_linear = nn.Linear(128, command_dim)
        self.log_std_linear = nn.Linear(128, command_dim)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(5.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)


    def forward(self, state):
        latent = self.backbone(state)

        mean = self.mean_linear(latent)
        log_std = self.log_std_linear(latent)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        '''
        state: shape = (batch_size, observation_dim=361)

        action: shape = (batch_size, command_dim)
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.backbone.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianActor, self).to(device) 
    


class Critic(nn.Module):
    def __init__(self, history_dim, observation_dim, kin_input_dim, kin_output_dim, n_history_frame, action_dim, *args, **kwargs):
        super(Critic, self).__init__(*args, **kwargs)

        # Q1
        self.backbone1 = BackBone(history_dim, observation_dim, kin_input_dim, kin_output_dim, 128, n_history_frame)
        self.action_embed1 = nn.Linear(action_dim, 128)
        self.v1 = nn.Linear(256, 1)

        # Q2
        self.backbone2 = BackBone(history_dim, observation_dim, kin_input_dim, kin_output_dim, 128, n_history_frame)
        self.action_embed2 = nn.Linear(action_dim, 128)
        self.v2 = nn.Linear(256, 1)


        self.apply(weights_init_)

    def forward(self, state, action):
        '''
        state: shape = (batch_size, observation_dim=361)
        action: shape = (batch_size, command_dim)

        v1: shape = (batch_size, 1)
        v2: shape = (batch_size, 1)
        '''
        state1 = self.backbone1(state)
        action1 = self.action_embed1(action)
        sa1 = torch.cat([state1, action1], dim=1)
        v1 = self.v1(sa1)

        state2 = self.backbone1(state)
        action2 = self.action_embed1(action)
        sa2 = torch.cat([state2, action2], dim=1)
        v2 = self.v2(sa2)

        return v1, v2
    
