import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import transforms
from fullsrgan import SRGANGenerator
from PIL import Image
import os


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.filenames = os.listdir(lr_dir)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        lr = Image.open(os.path.join(self.lr_dir, filename)).convert('RGB')
        hr = Image.open(os.path.join(self.hr_dir, filename)).convert('RGB')
        lr = lr.resize((1280, 720), Image.BICUBIC)
        hr = hr.resize((1920, 1080))
        return self.to_tensor(lr), self.to_tensor(hr)


lr_dir = 'data/lr720-1080'
hr_dir = 'data/hr'
output_dir = 'outputs_test'
os.makedirs(output_dir, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRGANGenerator().to(device)
model.load_state_dict(torch.load('srgan_generator3.pth'))
model.eval()


dataset = SRDataset(lr_dir, hr_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=False)


with torch.no_grad():
    for i, (lr_img, _) in enumerate(loader):
        lr_img = lr_img.to(device)
        sr_img = model(lr_img)
        save_image(sr_img, f"{output_dir}/sr_{i}.png")

print("Testing Done", output_dir)
