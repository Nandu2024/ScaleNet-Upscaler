from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
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
        lr = lr.resize((1280,720), Image.BICUBIC)
        hr = hr.resize((1920, 1080))
        return self.to_tensor(lr), self.to_tensor(hr)
