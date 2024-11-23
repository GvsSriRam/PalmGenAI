import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy
from torchvision import transforms
from torchvision.models import vgg16

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Rotate by up to 10 degrees
    transforms.ToTensor(),
])

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_size, weight_template_size):
        super(ConditionalDiffusionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Convolutional layer
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Convolutional layer
        self.bn2 = nn.BatchNorm2d(64)
        test_input = torch.randn(1, 1, 150, 150)
        test_output = self.conv2(F.max_pool2d(F.relu(self.bn1(self.conv1(test_input))), 2))
        correct_size = test_output.view(-1).shape[0]
        self.fc1 = nn.Linear(correct_size + weight_template_size, 1024) # Adjusted FC layer
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 64 * 75 * 75) # Output reshaped for deconvolution
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1) # Deconvolutional layer
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1) # Deconvolutional layer

    def forward(self, x, condition):
        # Concatenate the image and weight template (condition)
        print(x.shape)
        x = x.view(-1, 1, 150, 150) # Reshape for convolution
        print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        print(x.shape)
        x = F.max_pool2d(x, 2) # Downsampling
        print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        print(x.shape)
        x = F.max_pool2d(x, 3) # Downsampling
        print(x.shape)
        x = x.flatten(1)
        print(x.shape)
        # x = x.view(-1, 64 * 75 * 75) # Flatten for concatenation
        x = torch.cat((x, condition), dim=-1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        x = x.flatten(1)
        # x = x.view(-1, 64, 75, 75) # Reshape for deconvolution
        x = F.relu(self.bn5(self.deconv1(x)))
        x = F.interpolate(x, scale_factor=3) # Upsampling
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=2) # Upsampling
        x = x.view(-1, 150 * 150) # Flatten the output
        return x


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
        features_generated = self.vgg(x_generated.view(-1, 1, 150, 150))
        features_real = self.vgg(x_real.view(-1, 1, 150, 150))
        # Calculate L1 loss between features
        return F.l1_loss(features_generated, features_real)

    def loss_fn(self, x_noisy, t, x_start, condition):
        x_noisy = x_noisy.to(device)
        condition = condition.to(device)
        predicted_x_start = self.model(x_noisy, condition)
        perceptual_loss = self.perceptual_loss(predicted_x_start, x_start)
        mse_loss = F.mse_loss(predicted_x_start, x_start)  # Calculate MSE for monitoring
        return perceptual_loss, mse_loss

    def train(self, data_loader, optimizer, weight_templates, num_epochs=100, patience=10):
        best_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            train_loss = 0.0
            train_mse = 0.0
            for i, (x_start, idx) in enumerate(data_loader):
                x_start = x_start.to(device)
                x_start = transform(x_start)  # Apply transformations
                t = torch.randint(0, self.timesteps, (x_start.size(0),)).to(x_start.device)
                x_noisy = self.q_sample(x_start, t).to(x_start.device)
                condition = weight_templates[idx].to(x_start.device)

                loss, mse = self.loss_fn(x_noisy, t, x_start, condition)
                train_loss += loss.item()
                train_mse += mse.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation loop (add this)
            self.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            val_mse = 0.0
            with torch.no_grad():
                for i, (x_start, idx) in enumerate(val_data_loader):  # Use your validation data loader
                    t = torch.randint(0, self.timesteps, (x_start.size(0),)).to(x_start.device)
                    x_noisy = self.q_sample(x_start, t)
                    condition = weight_templates[idx].to(x_start.device)
                    loss, mse = self.loss_fn(x_noisy, t, x_start, condition)
                    val_loss += loss.item()
                    val_mse += mse.item()
            val_loss /= len(val_data_loader)
            val_mse /= len(val_data_loader)

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss/len(data_loader):.4f}, Train MSE: {train_mse/len(data_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}")
            
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

# Input image size and weight template size
input_size = 150 * 150 * 1  # Example for 64x64 RGB images
weight_template_size = 128  # Assuming your weight template is 256-dimensional

# Initialize model
model = ConditionalDiffusionModel(input_size, weight_template_size).to(device)
# Initialize optimizers
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.001)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001)

optimizer = optimizer_adam

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
trainer = DiffusionTrainer(model)

# Load images and weight templates
images = np.load("Datasets/IITD Palmprint V1/Preprocessed/Left/X_train.npy")  # Shape: (N, 64, 64, 3)
weight_templates = np.load(f"Datasets/IITD Palmprint V1/Preprocessed/Left/X_train_pca_{weight_template_size}.npy")  # Shape: (N, 256)
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
dataset = TensorDataset(images, torch.arange(images.shape[0]))  # Pass indices for weight templates
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(test_images, torch.arange(test_images.shape[0]))
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train the model
trainer.train(data_loader, optimizer, weight_templates, num_epochs=1000)

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
        alpha_bar_t_prev = torch.tensor(self.alpha_bar[t-1], dtype=torch.float32, device=x.device) if t > 0 else torch.tensor(1.0, dtype=torch.float32, device=x.device)

        noise = torch.randn_like(x)
        return predicted_x_start * torch.sqrt(alpha_bar_t_prev) + noise * torch.sqrt(1 - alpha_bar_t_prev)

    def sample(self, condition, img_shape=(150, 150, 1)):
        x = torch.randn((1, np.prod(img_shape))).to(condition.device)
        self.model.eval()
        with torch.no_grad():
            for t in reversed(range(self.timesteps)):
                x = self.p_sample(x, t, condition)
        return x.view(img_shape)

# Initialize sampler
sampler = DiffusionSampler(model)

# Load weight templates for the user
user_weight_template = weight_templates[0]
user_weight_template = user_weight_template.clone().detach().float().unsqueeze(0)
# Generate deepfake
generated_image = sampler.sample(user_weight_template)
generated_image = generated_image.detach().cpu().numpy()

# Reshape to original image dimensions and save or display
generated_image = generated_image.reshape(150, 150, 1)

# Displaying the original and generated images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(images[0].clone().detach().cpu().numpy().reshape(150, 150), cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

# Generated image
axes[1].imshow(generated_image.reshape(150, 150), cmap='gray')
axes[1].set_title("Generated Image")
axes[1].axis('off')

# Show the plot
plt.savefig('diffusion.png')
plt.close()
