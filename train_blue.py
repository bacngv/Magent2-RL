import torch
import torch.optim as optim
import random
from collections import deque
from magent2.environments import battle_v4
from torch_model import QNetwork

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TAU = 0.001  # Soft update factor for target network

# Initialize environment
env = battle_v4.env(map_size=45, render_mode=None)
obs_shape = env.observation_space("blue_0").shape
action_size = env.action_space("blue_0").n

# Initialize Q-network and target network
q_network = QNetwork(obs_shape, action_size)
target_network = QNetwork(obs_shape, action_size)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=LR)

# Replay buffer
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

def epsilon_greedy_policy(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)  # Random action
    with torch.no_grad():
        q_values = q_network(state)
        return torch.argmax(q_values).item()  # Best action

def train():
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32).permute(0, 3, 1, 2)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute Q-values and target Q-values using Double DQN
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
    with torch.no_grad():
        q_values_next = q_network(next_states)  # Q-values from current network
        max_q_values_next = target_network(next_states).gather(1, torch.argmax(q_values_next, dim=1).unsqueeze(1)).squeeze()
        target_q_values = rewards + GAMMA * max_q_values_next * (1 - dones)

    # Compute loss and optimize
    loss = torch.nn.functional.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def soft_update(target, source, tau=TAU):
    """ Soft update of target network's weights """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Training loop
# Training loop
epsilon = EPSILON_START
for episode in range(100):  # Number of episodes
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation

        if done:
            action = None  # This agent is done
        else:
            # Choose action using epsilon-greedy policy for blue agents
            if agent.startswith("blue"):  # Only apply to blue agents
                observation_tensor = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                action = epsilon_greedy_policy(observation_tensor, epsilon)
            else:
                action = env.action_space(agent).sample()  # Random action for red agents

        # Only add valid actions to the replay buffer
        if action is not None:
            replay_buffer.append((observation, action, reward, observation, done))

        env.step(action)  # Take the action

    if len(replay_buffer) > BATCH_SIZE:
        train()

    # Perform soft update of target network
    soft_update(target_network, q_network)

    # Save model checkpoint every episode
    torch.save(q_network.state_dict(), "blue.pt")
    
    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Print progress (optional)
    print(f"Episode {episode + 1}/100, Epsilon: {epsilon:.4f}")

env.close()
