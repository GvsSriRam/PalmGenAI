import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_template_size = 128
# Load the data
weight_templates = np.load("weight_templates.npy")
images = np.load(f"Datasets/IITD Palmprint V1/Preprocessed/Left/X_train_pca_{weight_template_size}.npy")

# Reshape images to (num_samples, 1, 150, 150)
images = images.reshape(-1, 1, 150, 150)

# Normalize images to [0, 1] if not already done
images = images.astype(np.float32) / 255.0 if images.max() > 1.0 else images

# Define the dataset
class PalmDataset(Dataset):
    def __init__(self, weight_templates, images):
        self.weight_templates = weight_templates
        self.images = images

    def __len__(self):
        return len(self.weight_templates)

    def __getitem__(self, idx):
        return self.weight_templates[idx], self.images[idx]

# Create the dataset and dataloader
dataset = PalmDataset(weight_templates, images)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.dec4 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(256, 128)
        self.dec2 = self.upconv_block(128, 64)
        self.dec1 = self.upconv_block(64, 32)

        # Output layer
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(in_channels, out_channels),
        )

    def forward(self, x, condition):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(torch.max_pool2d(enc1_out, 2))
        enc3_out = self.enc3(torch.max_pool2d(enc2_out, 2))
        enc4_out = self.enc4(torch.max_pool2d(enc3_out, 2))

        # Bottleneck
        bottleneck_out = self.bottleneck(torch.max_pool2d(enc4_out, 2))

        # Concatenate condition
        bottleneck_out = torch.cat([bottleneck_out, condition.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, bottleneck_out.shape[2], bottleneck_out.shape[3])], dim=1)

        # Decoder
        dec4_out = self.dec4(bottleneck_out)
        dec4_out = torch.cat([dec4_out, enc4_out], dim=1)
        dec3_out = self.dec3(dec4_out)
        dec3_out = torch.cat([dec3_out, enc3_out], dim=1)
        dec2_out = self.dec2(dec3_out)
        dec2_out = torch.cat([dec2_out, enc2_out], dim=1)
        dec1_out = self.dec1(dec2_out)
        dec1_out = torch.cat([dec1_out, enc1_out], dim=1)

        # Output
        out = torch.sigmoid(self.out(dec1_out))
        return out

# Create the model, optimizer, and loss function
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Define the noise schedule
def noise_schedule(t):
    return torch.cos(t * np.pi / 2)

# Training loop
epochs = 100
best_loss = float("inf")
patience = 5
epochs_without_improvement = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_idx, (weight_template, image) in enumerate(tqdm(dataloader)):
        weight_template = weight_template.to(device)
        image = image.to(device)

        # Sample time steps
        t = torch.rand(image.shape[0], device=device)

        # Forward process (add noise)
        noise = torch.randn_like(image)
        noisy_image = noise_schedule(t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * image + torch.sqrt(1 - noise_schedule(t)**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * noise

        # Predict noise
        predicted_noise = model(noisy_image, weight_template)

        # Calculate loss
        loss = criterion(predicted_noise, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Early stopping and model checkpointing
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "best_model.pth")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    # Reduce learning rate on plateau
    if epochs_without_improvement > 0 and epochs_without_improvement % 2 == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.5
        print(f"Reduced learning rate to {optimizer.param_groups[0]['lr']:.6f}")

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

# Generate an image from a random weight template
with torch.no_grad():
    idx = np.random.randint(len(weight_templates))
    weight_template = torch.tensor(weight_templates[idx], dtype=torch.float32).to(device)

    # Sample from standard normal distribution for initial noise
    x = torch.randn(1, 1, 150, 150).to(device)

    # Reverse diffusion process (sampling)
    for i in reversed(range(1000)):  # Number of diffusion steps
        t = torch.tensor([i / 1000], dtype=torch.float32).to(device)
        alpha_t = noise_schedule(t)
        
        # Predict noise
        predicted_noise = model(x, weight_template)
        
        # Update x based on the predicted noise
        x = (x - (1 - alpha_t**2) / alpha_t * predicted_noise) / alpha_t
        x = x + torch.sqrt(1 - alpha_t**2) * torch.randn_like(x)

    generated_image = x.cpu().numpy().squeeze()

# Display the generated image
plt.imshow(generated_image, cmap="gray")
plt.title("Generated Palm Image")
plt.show()