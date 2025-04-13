import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from fullsrgan import load_generator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


scale_to_model = {
    "180p → 720p (1280x720) 4x": {
        "weights": "best_srgan_generator2.pth",
        "lr_size": (320, 180),
        "scale": 4
    },
    "270p → 1080p (1920x1080) 4x": {
        "weights": "best_srgan_generator.pth",
        "lr_size": (480, 270),
        "scale": 4
    },
    "720p → 1080p (1920x1080) 1.5x": {
        "weights": "best_srgan_generator3.pth",
        "lr_size": (1280, 720),
        "scale": 1.5
    }
}


to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def upscale_with_choice(input_img, target_res):
    try:
        config = scale_to_model[target_res]
        weights = config["weights"]
        lr_size = config["lr_size"]
        scale = config["scale"]

        
        model = load_generator(weights_path=weights, upscale=scale, device=device)

        
        lr_img = input_img.resize(lr_size, Image.BICUBIC)
        lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

        
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        return to_pil(sr_tensor.squeeze().cpu().clamp(0, 1))

    except Exception as e:
        print(f"Error during upscaling: {e}")
        return f"Failed to upscale image. Reason: {e}"


demo = gr.Interface(
    fn=upscale_with_choice,
    inputs=[
        gr.Image(type="pil", label="Upload LR Image"),
        gr.Dropdown(
            choices=list(scale_to_model.keys()),
            value=list(scale_to_model.keys())[0],
            label="Target Resolution"
        )
    ],
    outputs=gr.Image(type="pil", label="Upscaled Output"),
    title="ScaleNet - AI Upscaler",
    description="Upload a low-resolution image and select your target resolution. ScaleNet will automatically choose the correct model and handle different architectures for you!"
)


if __name__ == "__main__":
    demo.launch()
