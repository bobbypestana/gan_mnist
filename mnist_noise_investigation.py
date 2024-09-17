import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import wandb

# Initialize Weights & Biases
wandb.init(project="gan-noise-investigation")

# Configuration for W&B
config = wandb.config
config.latent_size = 784
config.hidden_size = 256
config.image_size = 784
config.batch_size = 50
config.learning_rate = 0.0001
config.num_epochs = 200
config.noise_mean = 0.0  # Mean of the noise
config.noise_std = 1.0   # Standard deviation of the noise

# Check for GPU and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        z = torch.randn(config.batch_size, config.latent_size).to(device) * config.noise_std + config.noise_mean
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(config.batch_size, config.latent_size).to(device) * config.noise_std + config.noise_mean
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print and log progress
        if (i + 1) % (len(data_loader) // 4) == 0:
            elapsed_time = time.time() - start_time
            print(f'Epoch [{epoch}/{config.num_epochs}], Step [{i + 1}/{len(data_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                  f'D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}, '
                  f'Time Elapsed: {elapsed_time:.2f} sec')

            # Log metrics to W&B
            wandb.log({
                'Epoch': epoch,
                'D Loss': d_loss.item(),
                'G Loss': g_loss.item(),
                'D(x)': real_score.mean().item(),
                'D(G(z))': fake_score.mean().item(),
                'Time Elapsed': elapsed_time
            })

    # Save and visualize generated images
    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            grid = torchvision.utils.make_grid(fake_images, nrow=10, normalize=True)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.show()

# Save models
timestamp = time.time_ns()

torch.save(G.state_dict(), f'generator_{timestamp}.pth')
torch.save(D.state_dict(), f'discriminator_{timestamp}.pth')

# End W&B run
wandb.finish()
