import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import vgg16

# Hyperparameters
batch_size = 8
input_size = 150 * 150 * 1  # Example for 64x64 RGB images
weight_template_size = 64  # Assuming your weight template is 256-dimensional
lr = 1e-6

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your transformations
transform = transforms.Compose([
    transforms.ToPILImage('L'),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a   

        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633aefac1cf68ae434c8d13a48b2d3dc   

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(self, input_size, weight_template_size, n_channels=1, n_classes=1, bilinear=True):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // self.factor)

        self.fc_in_features = (512 // self.factor) * 16 * 16  # Calculate dynamically
        self.fc1 = nn.Linear(self.fc_in_features + weight_template_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.fc_in_features)

        self.up1 = Up(512, 256 // self.factor, bilinear)
        self.up2 = Up(256, 128 // self.factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, condition):
        x = x.view(-1, 1, 128, 128)  # Reshape for 128x128
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4_flat = x4.flatten(1)
        x_concat = torch.cat((x4_flat, condition), dim=-1)

        x = F.relu(self.fc1(x_concat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(-1, 512 // self.factor, 16, 16)  # Reshape for upsampling

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        logits = torch.sigmoid(logits)  # Sigmoid for output
        return logits.view(-1, 128 * 128)  # Flatten output


class DiffusionTrainer:
    def __init__(self, model, timesteps=10000):
        self.model = model
        self.timesteps = timesteps
        self.beta = np.linspace(1e-4, 0.02, timesteps)  # Linear schedule
        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        self.alpha_bar = torch.tensor(self.alpha_bar, dtype=torch.float32).to(device)

        # Initialize VGG model for perceptual loss
        self.vgg = vgg16(pretrained=True).features.to(device)
        # Modify the first layer to accept 1 channel
        self.vgg[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        for param in self.vgg.parameters():
            param.requires_grad = False

    def sample_noise(self, shape):
        return torch.randn(shape)

    def q_sample(self, x_start, t):
        x_start = x_start.to(device)
        noise = self.sample_noise(x_start.shape).to(x_start.device)

        # Correct broadcasting for alpha_bar_t
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)

        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise

    def perceptual_loss(self, x_generated, x_real):
        # Extract features from VGG16
        x_generated = torch.cat([x_generated.view(-1, 1, 128, 128)] * 3, dim=1)  # Reshape for VGG
        features_generated = self.vgg(x_generated)
        x_real = torch.cat([x_real.view(-1, 1, 128, 128)] * 3, dim=1)  # Reshape for VGG
        x_real = x_real.to(device)
        features_real = self.vgg(x_real)
        # Calculate L1 loss between features
        return F.l1_loss(features_generated, features_real)

    def loss_fn(self, x_noisy, t, x_start, condition):
        x_noisy = x_noisy.to(device)
        condition = condition.to(device)
        x_start = x_start.to(device)
        predicted_x_start = self.model(x_noisy, condition).to(device)
        if x_start.shape != predicted_x_start.shape:
            x_start = x_start.view(predicted_x_start.shape)

        mse_loss = F.mse_loss(predicted_x_start, x_start)  # Calculate MSE for monitoring
        return mse_loss

    def train(self, data_loader, optimizer, num_epochs=1000, patience=15):
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            train_mse = 0.0
            for i, (x_start, condition) in enumerate(data_loader):
                print(x_start.shape, condition.shape)
                x_start = x_start.to(device)
                x_start = transform(x_start)  # Apply transformations
                print(x_start.shape)
                t = torch.randint(0, self.timesteps, (x_start.size(0),)).to(x_start.device)
                x_noisy = self.q_sample(x_start, t).to(x_start.device)
                condition = condition.to(x_start.device)
                print(x_noisy.shape, x_start.shape, condition.shape)

                mse = self.loss_fn(x_noisy, t, x_start, condition)
                train_mse += mse.item()

                optimizer.zero_grad()
                mse.backward()  # Backpropagate MSE
                optimizer.step()

            # Validation loop
            self.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            val_mse = 0.0
            with torch.no_grad():
                for i, (x_start, condition) in enumerate(val_data_loader):  # Use your validation data loader
                    x_start = x_start.to(device)
                    x_start = transform(x_start)  # Apply transformations to validation data
                    t = torch.randint(0, self.timesteps, (x_start.size(0),)).to(x_start.device)
                    x_noisy = self.q_sample(x_start, t)
                    # condition = weight_templates[idx].to(x_start.device)
                    condition = condition.to(x_start.device)
                    mse = self.loss_fn(x_noisy, t, x_start, condition)
                    val_mse += mse.item()
            val_mse /= len(val_data_loader)
            val_loss = val_mse  # Use weighted loss for validation

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train MSE: {train_mse / len(data_loader):.4f}, "
                  f"Val MSE: {val_mse:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())  # Save best model weights
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(best_model_state)  # Load best model weights
                    break

            scheduler.step(val_loss)  # Update learning rate based on validation loss


# Initialize model
model = ConditionalUNet(input_size, weight_template_size).to(device)
# Initialize optimizers
optimizer_adam = optim.Adam(model.parameters(), lr=lr)
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=lr)
optimizer_adamw = optim.AdamW(model.parameters(), lr=lr)

optimizer = optimizer_adam

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
trainer = DiffusionTrainer(model)

# Load images and weight templates
images = np.load("Datasets/IITD Palmprint V1/Preprocessed/Left/X_train.npy")  # Shape: (N, 64, 64, 3)
weight_templates = np.load(
    f"Datasets/IITD Palmprint V1/Preprocessed/Left/X_train_pca_{weight_template_size}.npy")  # Shape: (N, 256)
test_images = np.load("Datasets/IITD Palmprint V1/Preprocessed/Left/X_test.npy")
test_weight_templates = np.load(f"Datasets/IITD Palmprint V1/Preprocessed/Left/X_test_pca_{weight_template_size}.npy")

# Preprocessing
images = torch.tensor(images).float().view(-1, input_size)  # Flatten images
weight_templates = torch.tensor(weight_templates).float()
test_images = torch.tensor(test_images).float().view(-1, input_size)
test_weight_templates = torch.tensor(test_weight_templates).float()

# Move data to the appropriate device
images = images.to(device)
weight_templates = weight_templates.to(device)
test_images = test_images.to(device)
test_weight_templates = test_weight_templates.to(device)

# Create dataset and dataloader
dataset = TensorDataset(images, weight_templates)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(test_images, test_weight_templates)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Train the model
trainer.train(data_loader, optimizer, num_epochs=1000)


class DiffusionSampler:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.beta = np.linspace(1e-4, 0.02, timesteps)  # Linear schedule
        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

    def p_sample(self, x, t, condition):
        predicted_x_start = self.model(x, condition)
        alpha_bar_t = torch.tensor(self.alpha_bar[t], dtype=torch.float32, device=x.device)
        alpha_bar_t_prev = torch.tensor(self.alpha_bar[t - 1], dtype=torch.float32,
                                        device=x.device) if t > 0 else torch.tensor(1.0, dtype=torch.float32,
                                                                                    device=x.device)

        noise = torch.randn_like(x)
        return predicted_x_start * torch.sqrt(alpha_bar_t_prev) + noise * torch.sqrt(1 - alpha_bar_t_prev)

    def sample(self, condition, img_shape=(128, 128, 1)):  # Updated img_shape
        condition = condition.to(device)
        condition = condition.unsqueeze(0)
        x = torch.randn((1, np.prod(img_shape))).to(condition.device)
        self.model.eval()
        with torch.no_grad():
            for t in reversed(range(self.timesteps)):
                x = self.p_sample(x, t, condition)
        return x.view(batch_size, *img_shape)


# Initialize sampler
sampler = DiffusionSampler(model)

# Load weight templates for the user
user_weight_template = weight_templates[0]
user_weight_template = user_weight_template.clone().detach().float().unsqueeze(0)

# Generate deepfake
generated_image = sampler.sample(user_weight_template)
generated_image = generated_image.detach().cpu().numpy()

# Reshape to original image dimensions and save or display
generated_image = generated_image.reshape(128, 128, 1)  # Reshape to 128x128

# Displaying the original and generated images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image (resized to 128x128 for comparison)
original_image = transforms.ToPILImage()(images[0].clone().detach().cpu()).resize((128, 128))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

# Generated image
axes[1].imshow(generated_image.reshape(128, 128), cmap='gray')
axes[1].set_title("Generated Image")
axes[1].axis('off')

# Show the plot
plt.savefig('diffusion.png')
plt.close()
