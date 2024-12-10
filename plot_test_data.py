import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(file_path, model):
    """
    This function takes a file path as input, reads the data from the file, and generates
    two histograms: one for Score and one for Frames.

    :param file_path: str, path to the input text file containing the data
    """
     # Initialize lists to store the data
    scores = []
    frames = []

    match model:
        case 'model1':
            model_name = 'stable baseline original environment - model 1'
        case 'model2':
            model_name = 'stable baseline stacked environment - model 2'
        case 'model3':
            model_name = 'self-coded unchanged reward - model 3'
        case 'model4':
            model_name = 'self-coded changed reward - model 4'

    # Read the data from the file
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            scores.append(float(parts[3]))
            frames.append(int(parts[5]))

    # Plot histogram for Scores
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=10, color='blue', edgecolor='black')
    plt.title(f'Histogram of Scores ({model_name})')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(f"plots/score histogram {model_name}.png")  # Save the plot
    plt.show()

    # Plot histogram for Frames
    plt.figure(figsize=(8, 6))
    plt.hist(frames, bins=10, color='green', edgecolor='black')
    plt.title(f'Histogram of Frames ({model_name})')
    plt.xlabel('Frames')
    plt.ylabel('Frequency')
    plt.savefig(f"plots/frames histogram {model_name}.png")  # Save the plot
    plt.show()

# Example usage

import matplotlib.pyplot as plt

def scatter_plot():
    """
    Creates a scatter plot of Score vs Frames for multiple datasets, each represented in a different color.
    """

    file_paths = ['data/stable_baselines_unstacked_frames_test.txt',
                  'data/self_unchanged_reward_test.txt', 
                  'data/self_changed_reward_test.txt']  
    labels = ['model 1', 'model 3', 'model 4']     
    colors = ['blue', 'green', 'red']                    
    plt.figure(figsize=(10, 6))
    
    for i, file_path in enumerate(file_paths):
        scores = []
        frames = []
        
        # Read the data from the file
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                scores.append(float(parts[3]))
                frames.append(int(parts[5]))
        
        # Scatter plot for this dataset
        plt.scatter(frames, scores, label=labels[i], color=colors[i], alpha=0.7)
    
    # Plot customization
    plt.title("Score vs Frames Scatter Plot (for different models)")
    plt.xlabel("Frames")
    plt.ylabel("Score")
    plt.legend() 
    plt.grid(True)
    plt.savefig(f"plots/scatter plot.png")  # Save the plot
    plt.show()

# Example usage


scatter_plot()


plot_histograms('data/stable_baselines_unstacked_frames_test.txt', model='model1')
plot_histograms('data/stable_baselines_stacked_frames_test.txt', model='model2')
plot_histograms('data/self_unchanged_reward_test.txt', model='model3')
plot_histograms('data/self_changed_reward_test.txt', model='model4')