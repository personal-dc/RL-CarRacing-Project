from stable_baselines_train import train_wrapper
from stable_baselines_test import run_eval

# train_wrapper(stack_frames=True)
run_eval(stack_frames=True)

## train stable-baselines agents with stacked and unstacked states
train_wrapper(stack_frames=False)
train_wrapper(stack_frames=True)

## test stable-baselines agents with stacked and unstacked states
run_eval(stack_frames=False)
run_eval(stack_frames=True)