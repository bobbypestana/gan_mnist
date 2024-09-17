import itertools
import multiprocessing
import os

# Define noise types
noise_types = ['normal', 'uniform', 'lognormal', 'exponential', 'gamma', 'poisson', 'random_binary']

# Define ranges for parameters
noise_params = {
    'normal': {'noise_std': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], 'noise_mean': [0.0, 1.0, -1.0, 5.0, -5.0]},
    'uniform': {'noise_min': [-1.0, 0.0], 'noise_max': [0.5, 1.0]},
    'lognormal': {'noise_std': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], 'noise_mean': [0.0, 1.0, -1.0, 5.0, -5.0]},
    'exponential': {'noise_lambda': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]},
    'gamma': {'noise_alpha': [1.0, 2.0, 3.0], 'noise_beta': [1.0, 2.0]},
    'poisson': {'noise_lambda': [1.0, 2.0, 5.0]},
    'random_binary': {}  # No parameters to vary for random_binary
}

# Function to execute all parameter combinations for a given job
def run_sequential_job(job_id, noise_types, noise_params):
    project_wandb = f'gan-noise-investigation-{job_id}'
    
    for noise_type in noise_types:
        param_keys = noise_params[noise_type].keys()
        if param_keys:  # If there are parameters for the noise type
            param_values = itertools.product(*noise_params[noise_type].values())
            
            for param_combination in param_values:
                params = {key: value for key, value in zip(param_keys, param_combination)}
                cmd = f"python gan_training.py --project_wandb {project_wandb} --noise_type {noise_type}"
                
                # Append each parameter to the command
                for param, value in params.items():
                    cmd += f" --{param} {value}"
                
                print(f"Running command: {cmd}")
                os.system(cmd)
        else:
            # If there are no parameters (e.g., random_binary)
            cmd = f"python gan_training.py --project_wandb {project_wandb} --noise_type {noise_type}"
            print(f"Running command: {cmd}")
            os.system(cmd)

# Number of parallel jobs
N = 7
first_suffix = 4

# Start parallel processes, each running through all parameter combinations sequentially
if __name__ == "__main__":
    processes = []
    
    # Launch N parallel jobs
    for i in range(N):
        p = multiprocessing.Process(target=run_sequential_job, args=(i+first_suffix, noise_types, noise_params))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
