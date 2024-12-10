from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.envs.box2d import CarRacing

def train_wrapper(time_steps=1000000, stack_frames = False):
    # Create the environment
    env = CarRacing()

    # Wrap the environment for stable baselines
    env = DummyVecEnv([lambda: env])

    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4) if stack_frames else env

    # Use the PPO algorithm with the CnnPolicy
    model = PPO("CnnPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=time_steps)

    # Save the model
    dir = "ppo_carracing_wrapper_env" if stack_frames else "ppo_car_racing_original"
    model.save(dir)