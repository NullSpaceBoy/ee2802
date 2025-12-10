import os
from bisect import bisect_right

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ImagePatchDataset(Dataset):
    def _get_number_of_tiles(self, dims: tuple[int, int]):
        width, height = dims
        return ((width + self.patch_size - 1) // self.patch_size) * (
            (height + self.patch_size - 1) // self.patch_size
        )

    def __init__(
        self,
        data_frame,
        working_dir,
        cache_enabled=False,
        patch_size=32,
        transform=ToTensor(),
        idx=False,
    ):
        self.idx = idx
        self.cache_enabled = cache_enabled
        self.cache = None
        self.working_dir = working_dir
        self.patch_size = patch_size
        self.transform = transform
        self.data_frame = data_frame
        self.data_frame["cum_tiles"] = (
            self.data_frame[["width", "height"]]
            .apply(self._get_number_of_tiles, axis=1)
            .cumsum()
        )

    def __len__(self):
        return self.data_frame["cum_tiles"].iloc[-1]

    def __getitem__(self, idx):
        img_idx = bisect_right(self.data_frame["cum_tiles"], idx)
        img_patch_id = idx - (
            self.data_frame["cum_tiles"].iloc[img_idx - 1] if img_idx != 0 else 0
        )

        if self.cache_enabled and self.cache is not None and self.cache[0] == img_idx:
            image = self.cache[1].copy()
            patches_per_row = self.cache[2]
        else:
            img_data = self.data_frame.iloc[img_idx]["photo_id"]
            image_path = os.path.join(self.working_dir, f"{img_data}.webp")
            image = Image.open(image_path)
            patches_per_row = (image.size[0] + self.patch_size - 1) // self.patch_size

            if self.cache_enabled:
                self.cache = (img_idx, image.copy(), patches_per_row)

        patch_x = img_patch_id % patches_per_row
        patch_y = img_patch_id // patches_per_row

        x_start = patch_x * self.patch_size
        y_start = patch_y * self.patch_size
        x_end = min(x_start + self.patch_size, image.size[0])
        y_end = min(y_start + self.patch_size, image.size[1])

        image = image.crop((x_start, y_start, x_end, y_end))
        pad_x = self.patch_size - image.size[0]
        pad_y = self.patch_size - image.size[1]
        image = ImageOps.expand(image, border=(0, 0, pad_x, pad_y), fill=(0, 0, 0))
        image = self.transform(image)

        if self.idx:
            return idx, image
        return image
