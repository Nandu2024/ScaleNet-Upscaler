import os
from PIL import Image
from torchvision import transforms

hr_dir = 'data/hr'
lr_dir = 'data/lr720-1080'
os.makedirs(lr_dir, exist_ok=True)

scale = 1.5
transform_hr = transforms.Resize((1920, 1080), interpolation=Image.BICUBIC)
transform_lr = transforms.Resize((1280, 720), interpolation=Image.BICUBIC)

for img_name in os.listdir(hr_dir):
    img_path = os.path.join(hr_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img_hr = transform_hr(img)
    img_lr = transform_lr(img_hr)
    img_lr.save(os.path.join(lr_dir, img_name))
