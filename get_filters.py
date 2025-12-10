import os

import torch
from PIL import Image
from torchvision.transforms.functional import to_grayscale, to_pil_image

from srcnn import SRCNN

device = torch.device("cpu")

if __name__ == "__main__":
    with torch.no_grad():
        os.makedirs("tmp/filters", exist_ok=True)
        model = SRCNN().to(device)
        model.load_state_dict(
            torch.load("tmp/models/model_20.tar", map_location=device)["model"]
        )
        model.eval()

        grid = Image.new("L", (79, 79), color=255)
        for i, filter in enumerate(model.conv1.weight):
            x = i % 8
            y = i // 8
            grid.paste(to_grayscale(to_pil_image(filter)), (x * 10, y * 10))
        grid.save("tmp/filters/filters.png")

        model.load_state_dict(
            torch.load("tmp/models/model2_5.tar", map_location=device)["model"]
        )
        model.eval()
        for i, filter in enumerate(model.conv1.weight):
            x = i % 8
            y = i // 8
            grid.paste(to_grayscale(to_pil_image(filter)), (x * 10, y * 10))
        grid.save("tmp/filters/filters2.png")
