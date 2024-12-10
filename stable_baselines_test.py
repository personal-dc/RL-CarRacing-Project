import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def create_env(stack_frames = False):
    env = gym.make('CarRacing-v3', render_mode='rgb_array')
    # Wrap the environment with DummyVecEnv
    env = DummyVecEnv([lambda: env])

    # use 4-stack env
    env = VecFrameStack(env, n_stack=1) if stack_frames else env

    return env

# Load the saved PPO model
def load_model(stack_frames):
    model_path = "param/ppo_carracing_wrapper_env.zip" if stack_frames else "param/ppo_carracing_original.zip"
    model = PPO.load(model_path)

    return model


def run_eval(stack_frames = False):
    eval_data = []
    env = create_env(stack_frames=stack_frames)
    model = load_model(stack_frames=stack_frames)


    data_dir = 'stable_baselines_stacked_frames_test.txt' if stack_frames else 'stable_baselines_unstacked_frames_test.txt'

    for ep in range(1000):
        # Evaluate the model
        obs= env.reset()
        total_reward = 0
        for frame in range(1000):  # Run for 1000 steps (adjust as needed)
            action, _states = model.predict(obs, deterministic=True)  # Predict actions
            
            # print(env.step(action))
            obs, reward, done, info = env.step(action)  # Step through the environment
            total_reward += reward
            env.render()  # Render the environment (optional)
            if done:
                print(f'Ep {ep} Score: {round(total_reward[0], 2)} Frames: {frame}')
                f = open(f'./data/{data_dir}', 'a')
                f.write(f'Ep {ep} Score: {round(total_reward[0], 2)} Frames: {frame}')
                f.write('\n')
                break
        
        eval_data.append((ep, total_reward[0], frame))
        
    # Close the environment
    env.close()
