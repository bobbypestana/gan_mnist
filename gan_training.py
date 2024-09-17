import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import wandb
import datetime as dt
import numpy as np
import scipy.stats as stats
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Train GAN with different noise types and parameters.")
parser.add_argument('--noise_type', type=str, default='normal', help='Type of noise distribution')
parser.add_argument('--noise_mean', type=float, default=0.0, help='Mean of the noise distribution')
parser.add_argument('--noise_std', type=float, default=1.0, help='Standard deviation of the noise distribution')
parser.add_argument('--noise_min', type=float, default=-1.0, help='Min value for uniform noise')
parser.add_argument('--noise_max', type=float, default=1.0, help='Max value for uniform noise')
parser.add_argument('--noise_lambda', type=float, default=1.0, help='Lambda for exponential or Poisson noise')
parser.add_argument('--noise_alpha', type=float, default=2.0, help='Alpha for gamma noise')
parser.add_argument('--noise_beta', type=float, default=1.0, help='Beta for gamma noise')
parser.add_argument('--project_wandb', type=str, default='gan-noise-investigation-test', help='Project name for W&B')
args = parser.parse_args()

# Initialize Weights & Biases
wandb.init(project=args.project_wandb)

# Configuration for W&B
config = wandb.config
config.latent_size = 64
config.hidden_size = 256
config.image_size = 784
config.batch_size = 50
config.learning_rate = 0.0002
config.num_epochs = 200
config.noise_type = args.noise_type
config.noise_mean = args.noise_mean
config.noise_std = args.noise_std
config.noise_min = args.noise_min
config.noise_max = args.noise_max
config.noise_lambda = args.noise_lambda
config.noise_alpha = args.noise_alpha
config.noise_beta = args.noise_beta


device_id = int(args.project_wandb.strip('gan-noise-investigation-')) % 2


# Check for GPU and set device
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# Load MNIST dataset
mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=config.batch_size, shuffle=True)

# Define Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc2(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Instantiate models and move to device
G = Generator(config.latent_size, config.hidden_size, config.image_size).to(device)
D = Discriminator(config.image_size, config.hidden_size).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=config.learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=config.learning_rate)


# Monobit Frequency Test: Check if the number of 1's and 0's are approximately equal.
def monobit_frequency_test(noise):
    n = len(noise)
    count_ones = np.sum(noise)
    S = abs(count_ones - (n - count_ones))
    p_value = stats.norm.cdf(S / np.sqrt(n))  # Normal distribution approximation
    return p_value

# Block Frequency Test: Check if blocks of the sequence have an equal number of 1's and 0's.
def block_frequency_test(noise, block_size=128):
    """
    Block Frequency Test: Check if blocks of the sequence have an equal number of 1's and 0's.

    Parameters:
        noise (torch.Tensor): Binary noise sequence.
        block_size (int): Size of each block to test.

    Returns:
        p_value (float): p-value of the test.
    """
    # Calculate the number of blocks
    num_blocks = len(noise) // block_size
    
    # Calculate the sum of 1's in each block
    block_sums = [np.sum(noise[i * block_size: (i + 1) * block_size]) for i in range(num_blocks)]
    
    # Convert block_sums to a NumPy array to allow element-wise operations
    block_sums = np.array(block_sums)
    
    # Perform the chi-squared test
    chi_squared = 4 * block_size * np.sum(((block_sums - block_size / 2) / block_size) ** 2)
    
    # Calculate the p-value from the chi-squared statistic
    p_value = stats.chi2.sf(chi_squared, num_blocks)  # Survival function
    
    return p_value


# Runs Test: Tests the randomness of a sequence by examining the number of runs.
def runs_test(noise):
    n = len(noise)
    count_ones = np.sum(noise)
    count_zeros = n - count_ones
    runs = 1 + np.sum(noise[1:] != noise[:-1])
    expected_runs = 2 * count_ones * count_zeros / n + 1
    variance_runs = 2 * count_ones * count_zeros * (2 * count_ones * count_zeros - n) / (n ** 2 * (n - 1))
    z = abs(runs - expected_runs) / np.sqrt(variance_runs)
    p_value = 2 * stats.norm.cdf(-z)  # Two-tailed test
    return p_value

# Longest Runs of Ones in a Block Test
def longest_runs_of_ones_test(noise, block_size=128):
    num_blocks = len(noise) // block_size
    longest_runs = [np.max(np.diff(np.where(np.concatenate(([0], noise[i * block_size:(i + 1) * block_size], [0])) == 0))) 
                    for i in range(num_blocks)]
    p_value = stats.chi2.sf(np.sum(longest_runs), num_blocks)
    return p_value

# Add more tests as necessary...

def noise_metrics(noise):
    """
    Calculate and log metrics for the generated noise.
    
    Metrics calculated:
        - Mean
        - Standard Deviation
        - Skewness
        - Kurtosis
        - Entropy
        - Range
        - Randomness Tests: Monobit Frequency, Block Frequency, Runs Test, Longest Runs of Ones
    
    Parameters:
        noise (torch.Tensor): The noise tensor to analyze.
    """
    noise_np = noise.cpu().numpy()

    # Statistical metrics
    mean = np.mean(noise_np)
    std = np.std(noise_np)
    skewness = np.mean((noise_np - mean) ** 3) / std**3
    kurtosis = np.mean((noise_np - mean) ** 4) / std**4 - 3
    noise_range = np.max(noise_np) - np.min(noise_np)
    
    hist, _ = np.histogram(noise_np, bins=100, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))

    # Convert noise to binary for randomness tests (e.g., threshold at 0.5)
    binary_noise = (noise_np > 0.5).astype(int)

    # Randomness tests
    p_monobit = monobit_frequency_test(binary_noise)
    p_block = block_frequency_test(binary_noise)
    p_runs = runs_test(binary_noise)
    p_longest_runs = longest_runs_of_ones_test(binary_noise)

    # Log metrics to Weights & Biases
    wandb.log({
        "Noise Mean": mean,
        "Noise Std": std,
        "Noise Skewness": skewness,
        "Noise Kurtosis": kurtosis,
        "Noise Range": noise_range,
        "Noise Entropy": entropy,
        "Monobit Frequency Test": p_monobit,
        "Block Frequency Test": p_block,
        "Runs Test": p_runs,
        "Longest Runs of Ones Test": p_longest_runs
    })


def generate_noise(batch_size, latent_size, noise_type='normal'):
    """
    Generates noise based on different distributions and logs metrics for the noise.

    Parameters:
        batch_size (int): The number of noise vectors to generate.
        latent_size (int): The size of each noise vector.
        noise_type (str): The type of distribution to sample from.

    Returns:
        torch.Tensor: The generated noise vector.
    """
    
    if noise_type == 'normal':
        z = torch.randn(batch_size, latent_size).to(device) * config.noise_std + config.noise_mean

    elif noise_type == 'uniform':
        z = torch.rand(batch_size, latent_size).to(device) * (config.noise_max - config.noise_min) + config.noise_min

    elif noise_type == 'exponential':
        z = torch.distributions.Exponential(config.noise_lambda).sample((batch_size, latent_size)).to(device)

    elif noise_type == 'lognormal':
        z = torch.distributions.LogNormal(config.noise_mean, config.noise_std).sample((batch_size, latent_size)).to(device)

    elif noise_type == 'gamma':
        z = torch.distributions.Gamma(config.noise_alpha, config.noise_beta).sample((batch_size, latent_size)).to(device)

    elif noise_type == 'poisson':
        z = torch.poisson(torch.full((batch_size, latent_size), config.noise_lambda)).to(device)

    elif noise_type == 'random_binary':
        z = torch.randint(0, 2, (batch_size, latent_size)).float().to(device)  # Binary random noise

    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    # # Log noise distribution type and parameters
    # wandb.log({
    #     "Noise Type": noise_type,
    #     "Noise Mean": config.noise_mean if noise_type in ['normal', 'lognormal'] else -100,
    #     "Noise Std": config.noise_std if noise_type in ['normal', 'lognormal'] else -100,
    #     "Noise Lambda": config.noise_lambda if noise_type in ['exponential', 'poisson'] else -100,
    #     "Noise Alpha": config.noise_alpha if noise_type == 'gamma' else -100,
    #     "Noise Beta": config.noise_beta if noise_type == 'gamma' else -100,
    #     "Noise Min": config.noise_min if noise_type == 'uniform' else -100,
    #     "Noise Max": config.noise_max if noise_type == 'uniform' else -100
    # })

    # Calculate and log noise metrics
    # noise_metrics(z)

    return z

# Define stopping criteria thresholds
stop_epochs_threshold = 10  # Stop if D(x) ≈ 1.00 and D(G(z)) ≈ 0.00 for this many epochs
no_change_threshold = 0.01  # X% change threshold (e.g., 1% -> 0.01)
no_change_epochs_threshold = 10  # Number of epochs for no change
# performance_decline_epochs_threshold = 5  # Stop if performance declines for this many epochs

# Initialize tracking variables
consecutive_epochs = 0
no_change_epochs = 0
previous_d_x_value = None
previous_d_gz_value = None
# performance_decline_epochs = 0

# Training loop
start_time = time.time()

for epoch in range(config.num_epochs):
    start_time = time.time()

    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(config.batch_size, -1).to(device)
        real_labels = torch.ones(config.batch_size, 1).to(device)
        fake_labels = torch.zeros(config.batch_size, 1).to(device)

        # Train Discriminator
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Generate noise with tunable parameters
        z = generate_noise(config.batch_size, config.latent_size, config.noise_type)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = generate_noise(config.batch_size, config.latent_size, config.noise_type)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print and log progress
        if (i + 1) % (len(data_loader) // 4) == 0:
            elapsed_time = time.time() - start_time

            d_x_value = real_score.mean().item()
            d_gz_value = fake_score.mean().item()

            print(f'Epoch [{epoch}/{config.num_epochs}], Step [{i + 1}/{len(data_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                  f'D(x): {d_x_value:.2f}, D(G(z)): {d_gz_value:.2f}, '
                  f'Time Elapsed: {elapsed_time:.2f} sec')

            noise_metrics(z)


            # Log metrics to W&B
            wandb.log({
                'Epoch': epoch,
                'D Loss': d_loss.item(),
                'G Loss': g_loss.item(),
                'D(x)': d_x_value,
                'D(G(z))': d_gz_value,
                'Time Elapsed': elapsed_time
            })

    # Criterion 1: Stop if D(x) ≈ 1.00 and D(G(z)) ≈ 0.00 for consecutive epochs
    if d_x_value >= 0.98 and d_gz_value <= 0.02:
        consecutive_epochs += 1
    else:
        consecutive_epochs = 0

    # Criterion 2: Stop if no significant change in D(x) and D(G(z)) for N epochs
    if previous_d_x_value is not None and previous_d_gz_value is not None:
        d_x_change = abs(d_x_value - previous_d_x_value) / max(abs(previous_d_x_value), 1e-8)
        d_gz_change = abs(d_gz_value - previous_d_gz_value) / max(abs(previous_d_gz_value), 1e-8)

        if d_x_change <= no_change_threshold and d_gz_change <= no_change_threshold:
            no_change_epochs += 1
        else:
            no_change_epochs = 0

        previous_d_x_value = d_x_value
        previous_d_gz_value = d_gz_value
    else:
        previous_d_x_value = d_x_value
        previous_d_gz_value = d_gz_value

    # # Criterion 3: Stop if performance goes down
    # if g_loss.item() > d_loss.item():  # Performance decline: G is failing against D
    #     performance_decline_epochs += 1
    # else:
    #     performance_decline_epochs = 0

    # Check stopping conditions
    if consecutive_epochs >= stop_epochs_threshold:
        print(f"Stopping training as D(x) ≈ 1.00 and D(G(z)) ≈ 0.00 for {stop_epochs_threshold} consecutive epochs.")
        break
    if no_change_epochs >= no_change_epochs_threshold:
        print(f"Stopping training as D(x) and D(G(z)) did not change by more than {no_change_threshold * 100}% "
              f"for {no_change_epochs_threshold} consecutive epochs.")
        break
    # if performance_decline_epochs >= performance_decline_epochs_threshold:
    #     print(f"Stopping training as performance is declining for {performance_decline_epochs_threshold} consecutive epochs.")
    #     break

    # # Save and visualize generated images
    # if (epoch + 1) % 20 == 0:
    #     with torch.no_grad():
    #         fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    #         grid = torchvision.utils.make_grid(fake_images, nrow=10, normalize=True)
    #         plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    #         plt.show()

# Save models with wandb id and timestamp
timestamp = dt.datetime.now().strftime("%y%m%d%H%M%S")
torch.save(G.state_dict(), f'models/generator_{wandb.run.id}_{timestamp}.pth')
torch.save(D.state_dict(), f'models/discriminator_{wandb.run.id}_{timestamp}.pth')


# End W&B run
wandb.finish()
