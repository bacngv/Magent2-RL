import torch.nn as nn
import torch


class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)

class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
            nn.Softmax(dim=-1),  # Output probabilities
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)


class MAAC(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        
        # CNN for processing observations
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
        )
        
        flatten_dim = 128  # Output dimension after pooling
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_shape), 
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(flatten_dim + action_shape, 256),  
            nn.ReLU(),
            nn.Linear(256, 1), 
        )

    def forward(self, observation):
        assert len(observation.shape) == 4, "Observation must be (batch_size, channels, height, width)"
        
        x = self.cnn(observation)  # Output shape: (batch_size, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128)
        action_probs = self.actor(x)  # Output shape: (batch_size, action_shape)
        
        return action_probs

    def evaluate(self, observation, action):
        """ Evaluate Q-value for given state-action pair """
        assert len(observation.shape) == 4, "Observation must be (batch_size, channels, height, width)"
        
        x = self.cnn(observation)
        x = x.view(x.size(0), -1)
        
        action_probs = self.actor(x)
        
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=action_probs.shape[1]).float().to(observation.device)
        state_action_value = self.critic(torch.cat([x, action_one_hot], dim=1))
        
        return state_action_value, action_probs
