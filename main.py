import os
import cv2
import torch
import imageio
from magent2.environments import battle_v4
from torch_model import QNetwork, PolicyNetwork


def load_model(model_path, model_class, observation_shape, action_shape):
    model = model_class(observation_shape, action_shape)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def save_gif(frames, gif_path, fps):
    duration = 1 / fps
    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"Saved GIF to {gif_path}")


def simulate(env, agents, vid_path, gif_path, max_steps=10000, fps=35):
    env.reset()
    frames = []

    for step in range(max_steps):
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                agent_team = agent.split("_")[0]
                model = agents.get(agent_team, None)

                if model:
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        if isinstance(model, PolicyNetwork):
                            action_probs = model(obs_tensor)
                            action = torch.distributions.Categorical(action_probs).sample().item()
                        elif isinstance(model, QNetwork):
                            q_values = model(obs_tensor)
                            action = torch.argmax(q_values, dim=1).item()
                else:
                    action = env.action_space(agent).sample()

            env.step(action)

            if agent == "blue_0":
                frames.append(env.render())

    height, width, _ = frames[0].shape
    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved video to {vid_path}")

    save_gif(frames, gif_path, fps)


if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array", max_cycles=10000)
    vid_dir = "video"
    gif_dir = "assets"
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    fps = 35

    red_model = load_model(
        "pretrained/red.pt",
        QNetwork,
        env.observation_space("red_0").shape,
        env.action_space("red_0").n,
    )
    blue_model = load_model(
        "pretrained/blue_final.pt",
        PolicyNetwork,
        env.observation_space("blue_0").shape,
        env.action_space("blue_0").n,
    )

    scenarios = {
        #"blue_vs_random": {"blue": blue_model},
        "blue_vs_red": {"blue": blue_model, "red": red_model},
    }

    for scenario_name, agents in scenarios.items():
        vid_path = os.path.join(vid_dir, f"{scenario_name}.mp4")
        gif_path = os.path.join(gif_dir, f"{scenario_name}.gif")
        simulate(env, agents, vid_path, gif_path, max_steps=10000, fps=fps)

    env.close()
