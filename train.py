import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from magent2.environments import battle_v4
from torch_model import QNetwork

# Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 5
MAX_EPISODES = 5
REPLAY_BUFFER_SIZE = 10000

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

def train_model(env, blue_q_network, red_q_network, optimizer, replay_buffer):
    epsilon = EPSILON_START
    steps_done = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                env.step(None)
                continue

            if agent.startswith("blue"):
                observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                if np.random.rand() < epsilon:
                    action = env.action_space(agent).sample()
                else:
                    with torch.no_grad():
                        q_values = blue_q_network(observation)
                    action = torch.argmax(q_values, dim=1).item()

                env.step(action)

                next_observation, reward, termination, truncation, _ = env.last()
                next_observation = torch.tensor(next_observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                # Adjust reward to encourage attacking and movement
                if info.get("attacked", False):
                    reward += 10.0  # High reward for successful attack
                elif action != 0:  # Encourage non-idle movement
                    reward += 0.5

                replay_buffer.push(
                    observation.squeeze(0).cpu().numpy(),
                    action,
                    reward,
                    next_observation.squeeze(0).cpu().numpy(),
                    termination or truncation,
                )

                episode_reward += reward

                if len(replay_buffer) >= BATCH_SIZE:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                    states, actions, rewards, next_states, dones = (
                        states.to(device), actions.to(device), rewards.to(device), 
                        next_states.to(device), dones.to(device)
                    )
                    q_values = blue_q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_q_values = blue_q_network(next_states).max(1)[0]
                    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
                    loss = nn.MSELoss()(q_values, target_q_values)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            elif agent.startswith("red"):
                observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = red_q_network(observation)
                action = torch.argmax(q_values, dim=1).item()
                env.step(action)

            else:
                env.step(env.action_space(agent).sample())

        epsilon = max(EPSILON_END, EPSILON_START - (steps_done / EPSILON_DECAY))
        steps_done += 1

        print(f"Episode {episode + 1}/{MAX_EPISODES}, Reward: {episode_reward}")

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            torch.save(blue_q_network.state_dict(), f"blue_agent_episode_{episode + 1}.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = battle_v4.env(
        map_size=45,
        minimap_mode=False,
        step_reward=0.2,
        dead_penalty=-0.2,
        attack_penalty=0.2,
        attack_opponent_reward=1.0,
        max_cycles=10000,
        extra_features=False
    )
    env.reset()

    red_q_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
    red_q_network.load_state_dict(torch.load("pretrained/red.pt", map_location=device))
    red_q_network.eval()

    blue_q_network = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    optimizer = optim.Adam(blue_q_network.parameters(), lr=LEARNING_RATE)

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    train_model(env, blue_q_network, red_q_network, optimizer, replay_buffer)

    torch.save(blue_q_network.state_dict(), "blue_agent_final.pt")
    env.close()
