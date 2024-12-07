import os
import torch
import numpy as np
from pettingzoo.magent import battle_v3
from algo import spawn_ai
from senarios.senario_battle import play

def run_battle_with_random_opponent(model_path, render_dir, map_size=45, max_steps=400, use_cuda=True):
    # Initialize the environment
    env = battle_v3.env(
        map_size=map_size,
        minimap_mode=True,
        step_reward=-0.005,
        dead_penalty=-0.1,
        attack_penalty=-0.1,
        attack_opponent_reward=0.2,
        max_cycles=max_steps,
        extra_features=True
    )
    handles = env.unwrapped.env.get_handles()

    # Load your trained model
    blue_model = spawn_ai('ac', env, handles[0], 'blue', max_steps, use_cuda)
    blue_model.load(model_path, step=0)

    # Create a random opponent model
    class RandomOpponent:
        def __init__(self, num_actions):
            self.num_actions = num_actions

        def act(self, **kwargs):
            batch_size = kwargs['obs'].shape[0]
            return np.random.randint(0, self.num_actions, batch_size)

    random_model = RandomOpponent(env.unwrapped.env.get_action_space(handles[1])[0])

    # Run the battle
    render_dir = os.path.abspath(render_dir)
    os.makedirs(render_dir, exist_ok=True)
    render_path = os.path.join(render_dir, "battle.gif")
    
    _, _, _, _, render_list = play(
        env=env,
        n_round=0,
        handles=handles,
        models=[blue_model, random_model],
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
    MODEL_PATH = "data/models/ac-0"  # Path to the trained model
    RENDER_DIR = "data"  # Directory to save the GIF

    run_battle_with_random_opponent(
        model_path=MODEL_PATH,
        render_dir=RENDER_DIR,
        map_size=45,
        max_steps=1000,
        use_cuda=True
    )
