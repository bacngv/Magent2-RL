import os
import cv2
import torch
from magent2.environments import battle_v4
from torch_model import QNetwork, PolicyNetwork


def load_model(model_path, model_class, observation_shape, action_shape):
    """
    Load a pre-trained model.
    """
    model = model_class(observation_shape, action_shape)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def simulate(env, agents, vid_path, max_steps=10000, fps=35):
    """
    Simulate an environment with specified agent policies and save the video.
    """
    env.reset()
    frames = []

    for step in range(max_steps):  # Run the simulation for the specified number of steps
        # Go through all agents in the environment
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None  # This agent is dead, no action
            else:
                agent_team = agent.split("_")[0]
                model = agents.get(agent_team, None)

                if model:  # Use the model to decide action
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        if isinstance(model, PolicyNetwork):  # On-policy agent
                            action_probs = model(obs_tensor)
                            action = torch.distributions.Categorical(action_probs).sample().item()
                        elif isinstance(model, QNetwork):  # Off-policy agent
                            q_values = model(obs_tensor)
                            action = torch.argmax(q_values, dim=1).item()
                else:  # Random action for other agents
                    action = env.action_space(agent).sample()

            env.step(action)

            # Record frames for the first agent of interest (blue_0 or red_0)
            # Instead of breaking the loop on agent death, just continue recording frames
            if agent == "blue_0" or agent == "red_0":  # Adjust as needed if you want other agents
                frames.append(env.render())

        # Check if all agents on both teams are dead and exit loop
        # You can also add conditions to stop the simulation early if you want.
        if all(termination or truncation for agent in env.agent_iter()):
            print("All agents terminated or truncated. Ending simulation early.")
            break

    # Save the video
    if frames:  # Only save if there are frames to write
        height, width, _ = frames[0].shape
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved video to {vid_path}")
    else:
        print("No frames to save. Video not created.")


if __name__ == "__main__":
    # Setup environment with a higher max_cycles
    env = battle_v4.env(map_size=45, render_mode="rgb_array", max_cycles=10000)
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 35

    # Load models
    red_model = load_model(
        "pretrained/red.pt",
        QNetwork,
        env.observation_space("red_0").shape,
        env.action_space("red_0").n,
    )
    blue_model = load_model(
        "pretrained/blue.pt",
        PolicyNetwork,
        env.observation_space("blue_0").shape,
        env.action_space("blue_0").n,
    )

    # Scenarios
    scenarios = {
        #"blue_vs_random": {"blue": blue_model},
        "blue_vs_red": {"blue": blue_model, "red": red_model},  # Blue agent vs. Red agent
    }

    # Simulate each scenario with a larger number of steps
    for scenario_name, agents in scenarios.items():
        vid_path = os.path.join(vid_dir, f"{scenario_name}.mp4")
        simulate(env, agents, vid_path, max_steps=10000, fps=fps)  # Set max_steps to increase video duration

    # Clean up
    env.close()