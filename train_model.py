import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from fullsrgan import SRGANGenerator, SRGANDiscriminator
from losses import VGGPerceptualLoss
from dataset_loader import SRDataset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import os

torch.backends.cudnn.benchmark = True  


epochs = 60
batch_size = 2
lr = 1e-4
lambda_mse = 1.0
lambda_adv = 1e-3
lambda_perc = 0.006
early_stopping_patience = 10
best_loss = float("inf")
patience_counter = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler()  


dataset = SRDataset(lr_dir='data/lr720-1080', hr_dir='data/hr')
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

generator = SRGANGenerator().to(device)
discriminator = SRGANDiscriminator().to(device)

mse_loss = nn.MSELoss()
adv_loss = nn.BCEWithLogitsLoss()  
vgg_loss = VGGPerceptualLoss().to(device)

opt_g = optim.Adam(generator.parameters(), lr=lr)
opt_d = optim.Adam(discriminator.parameters(), lr=lr)

log_g_losses, log_d_losses = [], []


for epoch in range(epochs):
    generator.train()
    discriminator.train()
    total_g_loss, total_d_loss = 0.0, 0.0

    for lr_img, hr_img in loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)

       
        with autocast():
            sr_img = generator(lr_img).detach()
            d_real = discriminator(hr_img)
            d_fake = discriminator(sr_img)

            real_labels = torch.ones_like(d_real).to(device)
            fake_labels = torch.zeros_like(d_fake).to(device)

            loss_d_real = adv_loss(d_real, real_labels)
            loss_d_fake = adv_loss(d_fake, fake_labels)
            d_loss = (loss_d_real + loss_d_fake) / 2

        opt_d.zero_grad()
        scaler.scale(d_loss).backward()
        scaler.step(opt_d)
        scaler.update()

        
        with autocast():
            sr_img = generator(lr_img)
            d_fake = discriminator(sr_img)

            loss_g_adv = adv_loss(d_fake, real_labels)
            loss_g_mse = mse_loss(sr_img, hr_img)
            loss_g_perc = vgg_loss(sr_img, hr_img)

            g_loss = lambda_mse * loss_g_mse + lambda_adv * loss_g_adv + lambda_perc * loss_g_perc

        opt_g.zero_grad()
        scaler.scale(g_loss).backward()
        scaler.step(opt_g)
        scaler.update()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        log_g_losses.append(g_loss.item())
        log_d_losses.append(d_loss.item())

    
    avg_g_loss = total_g_loss / len(loader)
    avg_d_loss = total_d_loss / len(loader)
    print(f"[Epoch {epoch+1}/{epochs}] G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")

    
    if avg_g_loss < best_loss:
        best_loss = avg_g_loss
        patience_counter = 0
        torch.save(generator.state_dict(), "best_srgan_generator3.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("🔥 Early stopping triggered")
            break


torch.save(generator.state_dict(), "srgan_generator3.pth")


plt.figure(figsize=(10, 5))
plt.plot(log_g_losses, label="Generator Loss")
plt.plot(log_d_losses, label="Discriminator Loss")
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.title("SRGAN Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()
