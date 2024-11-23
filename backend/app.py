# Set up necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as TF

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torch.nn.functional as F

# U-Net Generator Class
class UNetGenerator(nn.Module):
  def __init__(self):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)  # Update input channels to 3
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        # Final output layer (outputs 3 channels)
        self.final = nn.Conv2d(64, 3, kernel_size=1)  # Output 3 channels for RGB

  def conv_block(self, in_channels, out_channels):
      block = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
      )
      return block

  def forward(self, x):
      # Encoding path
      enc1 = self.enc1(x)  # Size: [B, 64, 256, 256]
      enc2 = self.enc2(F.max_pool2d(enc1, 2))  # Size: [B, 128, 128, 128]
      enc3 = self.enc3(F.max_pool2d(enc2, 2))  # Size: [B, 256, 64, 64]
      enc4 = self.enc4(F.max_pool2d(enc3, 2))  # Size: [B, 512, 32, 32]

      # Bottleneck
      bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

      # Decoding path
      dec4 = self.dec4(F.interpolate(bottleneck, scale_factor=2, mode='nearest'))
      dec3 = self.dec3(F.interpolate(dec4, scale_factor=2, mode='nearest'))
      dec2 = self.dec2(F.interpolate(dec3, scale_factor=2, mode='nearest'))
      dec1 = self.dec1(F.interpolate(dec2, scale_factor=2, mode='nearest'))

      # Final output layer
      output = self.final(dec1)  # Output should be [B, 3, 256, 256]

      return output

# Define PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=6):  # Update input_channels to 6
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # Now expecting 6 channels
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, img_A, img_B):
        # Concatenate along the channel dimension
        input = torch.cat((img_A, img_B), 1)
        return self.model(input)

# ----------------------------
# Loss Functions and Optimizers
# ----------------------------
criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()  # L1 loss (MAE)

# Dataset Class
class SARToColorDataset(Dataset):
    def __init__(self, data_dir, category, transform=None):
        self.data_dir = data_dir
        self.category = category
        self.transform = transform
        self.image_pairs = self._load_image_pairs()

    def _load_image_pairs(self):
        image_pairs = []
        s1_dir = os.path.join(self.data_dir, self.category, "s1")
        s2_dir = os.path.join(self.data_dir, self.category, "s2")

        s1_images = sorted(os.listdir(s1_dir))
        s2_images = sorted(os.listdir(s2_dir))

        for s1_img, s2_img in zip(s1_images, s2_images):
            image_pairs.append((os.path.join(s1_dir, s1_img), os.path.join(s2_dir, s2_img)))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        s1_img_path, s2_img_path = self.image_pairs[idx]

        # Load Sentinel-1 as grayscale and Sentinel-2 as RGB
        s1_img = Image.open(s1_img_path).convert("L")  # Load as single channel
        s2_img = Image.open(s2_img_path).convert("RGB")  # Load as RGB

        # Resize both images to 256x256
        resize = transforms.Resize((256, 256))
        s1_img = resize(s1_img)
        s2_img = resize(s2_img)

        # Convert to tensors
        s1_img = transforms.ToTensor()(s1_img)  # This will be [1, H, W]
        s2_img = transforms.ToTensor()(s2_img)   # This will be [3, H, W]

        # Replicate the single channel to create a 3-channel grayscale image
        s1_img = s1_img.repeat(3, 1, 1)  # From [1, H, W] -> [3, H, W]

        # Ensure both images have the same size
        assert s1_img.size() == s2_img.size(), f"Image sizes do not match: {s1_img.size()} vs {s2_img.size()}"

        return s1_img, s2_img

# ----------------------------
# Training and Testing Functions
# ----------------------------
def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, epochs=10):
    generator.train()
    discriminator.train()

    G_losses, D_losses = [], []

    for epoch in range(epochs):
        for i, (sar_img, color_img) in enumerate(dataloader):
            sar_img = sar_img.to(device)  # Keep this if you have to explicitly specify
            color_img = color_img.to(device)  # Keep this if you have to explicitly specify

            # Train Generator
            optimizer_G.zero_grad()
            gen_color = generator(sar_img)
            validity = discriminator(sar_img, gen_color)

            g_loss = criterion_GAN(validity, torch.ones_like(validity).to(device)) + criterion_pixelwise(gen_color, color_img)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion_GAN(discriminator(sar_img, color_img), torch.ones_like(validity).to(device))
            fake_loss = criterion_GAN(discriminator(sar_img, gen_color.detach()), torch.zeros_like(validity).to(device))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Store losses
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

        print(f"Epoch {epoch + 1}/{epochs} | Generator Loss: {g_loss.item()} | Discriminator Loss: {d_loss.item()}")

    return G_losses, D_losses

def test_and_visualize(generator, dataloader):
    generator.eval()
    with torch.no_grad():
        for i, (sar_img, color_img) in enumerate(dataloader):
            sar_img = sar_img.to(device)
            color_img = color_img.to(device)  # Move color_img to the same device as sar_img

            gen_color = generator(sar_img)

            # Calculate accuracy
            accuracy = calculate_accuracy(gen_color, color_img)

            # Visualization
            plt.subplot(1, 3, 1)
            plt.title("SAR Image")
            plt.imshow(sar_img[0].cpu().numpy().transpose(1, 2, 0), cmap="gray")

            plt.subplot(1, 3, 2)
            plt.title("Generated Color")
            plt.imshow(gen_color[0].cpu().numpy().transpose(1, 2, 0))

            plt.subplot(1, 3, 3)
            plt.title("Ground Truth")
            plt.imshow(color_img[0].cpu().numpy().transpose(1, 2, 0))

            plt.suptitle(f'Accuracy: {accuracy:.2f}%')
            plt.show()

            break  # Only visualize one batch


# Function to calculate accuracy
def calculate_accuracy(pred, target):
    pred = (pred > 0.5).float()  # Convert to binary
    target = (target > 0.5).float()  # Convert to binary
    correct = (pred == target).sum()
    total = target.numel()
    return (correct / total) * 100

if __name__ == "__main__":
    # Set the data directory (e.g., Google Drive path)
    data_dir = "/content/drive/MyDrive/SIH 2024/SIH 2024 data"

    categories = ["urban"]  # Order of categories to train on

    # Initialize models
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for category in categories:
        print(f"Training on category: {category}")
        # Initialize dataset and dataloader for the current category
        dataset = SARToColorDataset(data_dir, category, transform=None)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

        # Train the model
        G_losses, D_losses = train(generator, discriminator, dataloader, optimizer_G, optimizer_D, epochs=5)

        # Test and visualize results
        test_and_visualize(generator, dataloader)

        # Optionally, you can save the model after each category
        torch.save(generator.state_dict(), f"{category}_generator.pth")
        torch.save(discriminator.state_dict(), f"{category}_discriminator.pth")