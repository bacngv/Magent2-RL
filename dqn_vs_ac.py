import os
import torch
import numpy as np
from magent2.environments import battle_v4 
from algo import spawn_ai
from senarios.senario_battle import play
from torch_model import QNetwork

def run_battle_with_red_opponent(ac_model_path, red_model_path, render_dir, map_size=45, max_steps=400, use_cuda=True):
    # Initialize the environment
    env = battle_v4.env(map_size=map_size, render_mode="rgb_array")
    handles = env.unwrapped.env.get_handles()

    # Load your trained AC model
    blue_model = spawn_ai('ac', env, handles[0], 'blue', max_steps, use_cuda)
    blue_model.load(ac_model_path, step=50)  # Load at step 50 as requested

    # Load the pretrained Q-learning model
    q_network = QNetwork(
        env.observation_space("red_0").shape, 
        env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load(red_model_path, weights_only=True, map_location="cpu")
    )

    # Create a wrapper for the Q-learning model to match the play() function interface
    class QNetworkWrapper:
        def __init__(self, q_network):
            self.q_network = q_network
            self.num_actions = q_network.network[-1].out_features

        def act(self, obs, feature=None, prob=None, eps=0):
            # Add batch dimension if needed
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
            
            # Epsilon-greedy exploration
            if np.random.random() < eps:
                return np.random.randint(0, self.num_actions, obs.shape[0])
            
            # Use Q-network for action selection
            with torch.no_grad():
                q_values = self.q_network(obs)
                return torch.argmax(q_values, dim=1).numpy()

    red_model = QNetworkWrapper(q_network)

    # Run the battle
    render_dir = os.path.abspath(render_dir)
    os.makedirs(render_dir, exist_ok=True)
    render_path = os.path.join(render_dir, "battle.gif")

    _, _, _, _, render_list = play(
        env=env,
        n_round=0,
        handles=handles,
        models=[blue_model, red_model],
        print_every=50,
        eps=1.0,
        render=True,
        train=False,
        cuda=use_cuda
    )

    # Save the battle as a GIF
    if render_list:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        print(f"[*] Saving render to {render_path}...")
        clip = ImageSequenceClip(render_list, fps=20)
        clip.write_gif(render_path, fps=20, verbose=False)
        print("[*] Render saved!")

# Usage example
if __name__ == "__main__":
    AC_MODEL_PATH = "data/models/ac-0"  # Path to the AC trained model
    RED_MODEL_PATH = "red.pt"  # Path to the pretrained Q-learning model
    RENDER_DIR = "data"  # Directory to save the GIF

    run_battle_with_red_opponent(
        ac_model_path=AC_MODEL_PATH,
        red_model_path=RED_MODEL_PATH,
        render_dir=RENDER_DIR,
        map_size=45,
        max_steps=1000,
        use_cuda=True
    )
