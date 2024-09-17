import itertools
import subprocess
import os

# Define noise types
noise_types = ['normal', 'uniform','lognormal', 'exponential', 'gamma', 'poisson', 'random_binary'] # 'normal', 'uniform', 'lognormal', 'exponential', 'gamma', 'poisson', 'random_binary']

# Define ranges for parameters
noise_params = {
    # 'normal': {'noise_std': [0.1], 'noise_mean': [0.0]},
    'normal': {'noise_std': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], 'noise_mean': [0.0, 1.0, -1.0, 5.0, -5.0]},
    'uniform': {'noise_min': [-1.0, 0.0], 'noise_max': [0.5, 1.0]},
    'lognormal': {'noise_std': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], 'noise_mean': [0.0, 1.0, -1.0, 5.0, -5.0]},
    'exponential': {'noise_lambda': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]},
    'gamma': {'noise_alpha': [1.0, 2.0, 3.0], 'noise_beta': [1.0, 2.0]},
    'poisson': {'noise_lambda': [1.0, 2.0, 5.0]},
    'random_binary': {}  # No parameters to vary for random_binary
}

project_wandb = 'gan-noise-investigation-11'

# Generate commands for each noise type with all combinations of parameter values
for noise_type in noise_types:
    param_keys = noise_params[noise_type].keys()
    if param_keys:  # If there are parameters for the noise type
        param_values = itertools.product(*noise_params[noise_type].values())
        
        for param_combination in param_values:
            params = {key: value for key, value in zip(param_keys, param_combination)}
            # Create the command
            cmd = f"python gan_training.py --project_wandb {project_wandb} --noise_type {noise_type}"
            
            # Append each parameter to the command
            for param, value in params.items():
                cmd += f" --{param} {value}"
            
            # print(cmd)

            print(f"Running command: {cmd}")
            os.system(cmd)
            # Run the command using subprocess
            # subprocess.run(cmd, shell=True)
    else:
        # If there are no parameters (e.g., random_binary)
        cmd = f"python gan_training.py --project_wandb {project_wandb}--noise_type {noise_type}"
        # print(cmd)

        print(f"Running command: {cmd}")
        os.system(cmd)
        # Run the command using subprocess
        # subprocess.run(cmd, shell=True)