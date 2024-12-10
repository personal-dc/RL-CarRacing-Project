from self_train import train_agent
from self_test import run_test

## train self-coded agents with changed and unchanged reward functions
train_agent(changed_reward=True)
train_agent(changed_reward=False)

## test self-coded agents with changed and unchanged reward function
## NOTE: the testing environment has the original (unchanged) reward function
## the boolean is to help with loading the correct model
run_test(changed_reward=True)
run_test(changed_reward=False)