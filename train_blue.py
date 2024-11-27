import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from magent2.environments import battle_v4
from torch_model import PolicyNetwork

def train_model(env, blue_policy_network, optimizer, args):
    for episode in range(args.max_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        episode_reward = 0
        entropy_term = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                env.step(None)
                continue

            if agent.startswith("blue"):
                # Normalize observation
                observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

                # Policy sampling with exploration
                action_probs = blue_policy_network(observation)
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()

                log_prob = action_distribution.log_prob(action)
                log_probs.append(log_prob)
                entropy_term += action_distribution.entropy().mean()

                rewards.append(reward)
                env.step(action.item())
                episode_reward += reward

            elif agent.startswith("red"):
                # Red agents act randomly
                action = env.action_space(agent).sample()
                env.step(action)
            else:
                env.step(env.action_space(agent).sample())

        # Compute discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + args.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Compute policy loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Include entropy for exploration regularization
        loss = policy_loss - args.entropy_coef * entropy_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{args.max_episodes}, Reward: {episode_reward:.2f}")

        if (episode + 1) % args.save_interval == 0:
            model_path = f"pretrained/blue_episode_{episode + 1}.pt"
            torch.save(blue_policy_network.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # Save final model
    torch.save(blue_policy_network.state_dict(), "pretrained/blue_final.pt")
    print("Training complete. Final model saved to pretrained/blue_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a REINFORCE agent for battle_v4 environment")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--max_episodes", type=int, default=500, help="Maximum number of training episodes")
    parser.add_argument("--save_interval", type=int, default=50, help="Save model every n episodes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    env.reset()

    blue_policy_network = PolicyNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    optimizer = optim.Adam(blue_policy_network.parameters(), lr=args.learning_rate)

    train_model(env, blue_policy_network, optimizer, args)

    env.close()
