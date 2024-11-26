import torch.nn as nn
import torch


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
