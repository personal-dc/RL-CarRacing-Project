from sys import prefix
import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_data(reward_changed = True):
    """
    This function reads data from a file, loads it into a DataFrame, and plots:
    1. Episode vs Reward
    2. Episode vs Moving Average
    3. Episode vs both Reward and Moving Average (on a single plot)
    
    Each plot is shown one at a time.
    
    :param reward_changed: bool, is the data being plotted with changed reward function or not
    """

    prefix = '' if reward_changed else 'un'
    file_path = f'data/self_{prefix}changed_reward_train.txt'

    # Load the data into a DataFrame
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the tuple values
            values = line.strip().strip("()").split(", ")
            episode = int(values[0])
            reward = float(values[1])
            moving_avg = float(values[2])
            data.append((episode, reward, moving_avg))

    df = pd.DataFrame(data, columns=["Episode", "Reward", "Moving Average"])

    # Plot: Reward and Moving Average on the same plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["Episode"], df["Reward"], label="Reward", color="blue")
    plt.plot(df["Episode"], df["Moving Average"], label="Moving Average", color="orange")
    plt.title(f"Episode vs Reward and Moving Average ({prefix}changed reward)")
    plt.xlabel("Episode")
    plt.ylabel("Values")
    plt.legend()
    plt.savefig(f"plots/train_{prefix}changed_reward.png")  # Save the plot
    plt.show()

plot_reward_data(reward_changed=True)
plot_reward_data(reward_changed=False)
