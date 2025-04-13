import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class SRGANGenerator(nn.Module):
    def __init__(self, num_res_blocks=16, upscale=4):
        super().__init__()
        self.upscale = upscale

        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])

        self.middle_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        
        if upscale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        elif upscale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        else:
            
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upscale, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.PReLU()
            )

        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.res_blocks(x1)
        x3 = self.middle_conv(x2)
        x4 = x1 + x3
        x5 = self.upsample(x4)
        out = self.output_conv(x5)
        return (out + 1.0) / 2.0  


class SRGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            block(64, 64, 2),
            block(64, 128, 1),
            block(128, 128, 2),
            block(128, 256, 1),
            block(256, 256, 2),
            block(256, 512, 1),
            block(512, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)  # No sigmoid
        )

    def forward(self, x):
        return self.net(x)
def load_generator(weights_path, upscale=4, device="cpu"):
    try:
        model = SRGANGenerator(upscale=upscale).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Loaded with SRGANGenerator (upscale={upscale})")
        return model.eval()

    except Exception as e:
        print(f"⚠️ New model failed to load. Falling back to SRGANGeneratorOld.\nReason:\n{e}")

        class SRGANGeneratorOld(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_conv = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=9, padding=4),
                    nn.PReLU()
                )
                self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])
                self.middle_conv = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64)
                )
                self.upsample = nn.Sequential(
                    nn.Conv2d(64, 256, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU()
                )
                self.output_conv = nn.Sequential(
                    nn.Conv2d(64, 3, kernel_size=9, padding=4),
                    nn.Tanh()
                )

            def forward(self, x):
                x1 = self.input_conv(x)
                x2 = self.res_blocks(x1)
                x3 = self.middle_conv(x2)
                x4 = x1 + x3
                x5 = self.upsample(x4)
                out = self.output_conv(x5)
                return (out + 1.0) / 2.0

        model = SRGANGeneratorOld().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Loaded with SRGANGeneratorOld")
        return model.eval()
