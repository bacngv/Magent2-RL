import os
import cv2
import torch
import imageio
from magent2.environments import battle_v4
from torch_model import QNetwork
from MADDPG import MAAC


def load_model_red(model_path, observation_shape, action_shape):
    model = QNetwork(observation_shape, action_shape)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()
    return model

def load_model_blue(model_path, observation_shape, action_shape):
    model = MAAC(observation_shape, action_shape)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()
    return model


def mp4_to_gif(mp4_path, gif_path, fps=35):
    reader = cv2.VideoCapture(mp4_path)
    frames = []
    
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    reader.release()
    
    imageio.mimsave(gif_path, frames, duration=1/fps)
    print(f"Converted {mp4_path} to {gif_path}")


if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    vid_dir = "video"
    assets_dir = "assets"
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    fps = 35

    # Load models
    red_q_network = load_model_red("pretrained/red.pt", env.observation_space("red_0").shape, env.action_space("red_0").n)
    blue_q_network = load_model_blue("pretrained/blue.pt", env.observation_space("blue_0").shape, env.action_space("blue_0").n)

    # Scenario 1: Blue vs Random
    frames = []
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            if agent.startswith("blue"):
                observation = torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                with torch.no_grad():
                    q_values = blue_q_network(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                action = env.action_space(agent).sample()  # Random action for non-blue agents

        env.step(action)

        if agent == "blue_0":
            frames.append(env.render())

    blue_vs_random_mp4 = os.path.join(vid_dir, "blue_vs_random.mp4")
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        blue_vs_random_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording Blue vs Random")

    mp4_to_gif(blue_vs_random_mp4, os.path.join(assets_dir, "blue_vs_random.gif"), fps)

    # Scenario 2: Blue vs Pretrained Red
    frames = []
    env.reset()

    # Cần phải xử lý toàn bộ vòng lặp cho mỗi agent
    done = False  # Dùng biến này để kiểm tra xem môi trường đã kết thúc chưa

    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            # Xử lý khi agent đã kết thúc hoặc bị cắt ngắn
            if termination or truncation:
                action = None  # Không có hành động khi agent đã chết
            else:
                if agent.startswith("red"):
                    observation = torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                    with torch.no_grad():
                        q_values = red_q_network(observation)
                    action = torch.argmax(q_values, dim=1).numpy()[0]
                elif agent.startswith("blue"):
                    observation = torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                    with torch.no_grad():
                        q_values = blue_q_network(observation)
                    action = torch.argmax(q_values, dim=1).numpy()[0]
                else:
                    action = env.action_space(agent).sample()  # Random action cho agent khác

            # Gọi bước đi của agent
            env.step(action)

            # Nếu agent blue_0 thì lưu lại frame cho video
            if agent == "blue_0":
                frames.append(env.render())

        # Kiểm tra nếu tất cả agent đã terminated hoặc truncated
        done = all(env.last()[1] or env.last()[2] for _ in env.agent_iter())

    # Sau khi vòng lặp kết thúc, ghi lại video
    blue_vs_red_mp4 = os.path.join(vid_dir, "blue_vs_red.mp4")
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        blue_vs_red_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

    print("Done recording Blue vs Pretrained Red")

    # Chuyển đổi video MP4 thành GIF
    mp4_to_gif(blue_vs_red_mp4, os.path.join(assets_dir, "blue_vs_red.gif"), fps)



    # Scenario 3: Blue vs Blue (both blue agents use blue.pt)
    frames = []
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            if agent.startswith("blue"):
                observation = torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                with torch.no_grad():
                    q_values = blue_q_network(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                action = env.action_space(agent).sample()  # Random action for non-blue agents

        env.step(action)

        if agent == "blue_0":
            frames.append(env.render())

    blue_vs_blue_mp4 = os.path.join(vid_dir, "blue_vs_blue.mp4")
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        blue_vs_blue_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording Blue vs Blue")

    mp4_to_gif(blue_vs_blue_mp4, os.path.join(assets_dir, "blue_vs_blue.gif"), fps)

    env.close()
