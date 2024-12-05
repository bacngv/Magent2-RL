import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from magent2.environments import battle_v4
from torch_model import PolicyNetwork, QNetwork


def train_model(env, blue_policy_network, red_q_network, optimizer, args):
    all_rewards = []
    for episode in range(args.max_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        episode_reward = 0

        # Normalize observations (running mean/variance could be added here)
        def preprocess_observation(obs):
            return torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            reward = np.clip(reward, -1, 1)  # Clip reward to range [-1, 1]

            if termination or truncation:
                env.step(None)
                continue

            if agent.startswith("blue"):
                # Custom reward function to encourage aggression
                reward = custom_reward(observation, reward, agent)

                obs_tensor = preprocess_observation(observation)

                # Policy sampling
                action_probs = blue_policy_network(obs_tensor)
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()

                log_prob = action_distribution.log_prob(action)
                log_probs.append(log_prob)
                rewards.append(reward)

                env.step(action.item())
                episode_reward += reward

            elif agent.startswith("red"):
                obs_tensor = preprocess_observation(observation)
                with torch.no_grad():
                    q_values = red_q_network(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()
                env.step(action)
            else:
                env.step(env.action_space(agent).sample())

        # Calculate policy gradients and update
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + args.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        # Normalize rewards for better numerical stability
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(blue_policy_network.parameters(), max_norm=1.0)

        optimizer.step()

        all_rewards.append(episode_reward)

        # Logging
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(all_rewards[-args.log_interval:])
            print(
                f"Episode {episode + 1}/{args.max_episodes}, "
                f"Average Reward (Last {args.log_interval}): {avg_reward:.2f}, "
                f"Loss: {policy_loss.item():.4f}"
            )

        # Save model periodically
        if (episode + 1) % args.save_interval == 0:
            save_path = f"pretrained/blue_agent_episode_{episode + 1}.pt"
            torch.save(blue_policy_network.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    # Final model save
    final_save_path = "pretrained/blue.pt"
    torch.save(blue_policy_network.state_dict(), final_save_path)
    print(f"Training completed. Final model saved to {final_save_path}")


# Custom reward function to encourage aggression
def custom_reward(observation, reward, agent, blue_agent_proximity_bonus=0.3):
    if agent.startswith("blue") and is_near_opponent(observation):  # Check proximity to opponent
        reward += blue_agent_proximity_bonus
    return reward

def is_near_opponent(observation, proximity_threshold=0.1):
    # Assuming the observation is a numpy array or tensor, with the presence information for the other team
    # team_1_presence is a channel indicating the presence of the opponent team
    other_team_presence = observation[3]  # Adjust the index according to your observation structure
    # We assume the other team's presence is a 2D grid. We check if any cells have non-zero values.
    
    return np.any(other_team_presence > proximity_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a REINFORCE agent for battle_v4 environment")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--max_episodes", type=int, default=20, help="Maximum number of training episodes")
    parser.add_argument("--save_interval", type=int, default=50, help="Save model every n episodes")
    parser.add_argument("--log_interval", type=int, default=10, help="Log average reward every n episodes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment and models
    env = battle_v4.env(
        map_size=45,
        minimap_mode=False,
        step_reward=-0.01,  # Increased to encourage more action towards opponents
        dead_penalty=-1,  # Increased penalty for dying to discourage passive behavior
        attack_penalty=-0.002,  # Lowered to reduce the punishment for attacking
        attack_opponent_reward=10.0,  # Significantly increased reward for attacking opponents
        max_cycles=10000,
        extra_features=False
    )
    env.reset()

    red_q_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
    red_q_network.load_state_dict(torch.load("pretrained/red.pt", map_location=device))
    red_q_network.eval()

    blue_policy_network = PolicyNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    optimizer = optim.Adam(blue_policy_network.parameters(), lr=args.learning_rate)

    # Train the blue agent
    train_model(env, blue_policy_network, red_q_network, optimizer, args)

    env.close()