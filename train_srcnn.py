import os
import sys

import imagesize
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)
from tqdm import tqdm

from img_patch_dataset import ImagePatchDataset
from srcnn import SRCNN
from srcnn_preprocessor import PreprocessingModule


class EarlyStop:
    def __init__(self):
        super().__init__()
        self.patience = 10
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def state_dict(self):
        return {
            "patience": self.patience,
            "counter": self.counter,
            "best_loss": self.best_loss,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict["patience"]
        self.counter = state_dict["counter"]
        self.best_loss = state_dict["best_loss"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, preprocessor, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    for hr_batch in tqdm(
        train_loader,
        smoothing=0,
        desc=f"Epoch {epoch}",
        postfix={"LR": optimizer.param_groups[0]["lr"]},
    ):
        hr_batch = hr_batch.to(device)
        sr_batch = preprocessor(hr_batch)
        optimizer.zero_grad()
        sr_batch = model(sr_batch)
        loss = criterion(sr_batch, hr_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * sr_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def val_epoch(model, preprocessor, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for hr_batch in tqdm(val_loader, smoothing=0):
            hr_batch = hr_batch.to(device)
            sr_batch = preprocessor(hr_batch)
            sr_batch = model(sr_batch)
            loss = criterion(sr_batch, hr_batch)
            total_loss += loss.item() * sr_batch.size(0)
        avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_srcnn.py <working_dir>")
        sys.exit(1)

    os.makedirs(os.path.join("tmp", "models"), exist_ok=True)

    start_epoch = 1
    if len(sys.argv) == 3:
        start_epoch = int(sys.argv[2])

    working_dir = os.path.join(sys.argv[1], "train")
    df = pd.read_csv(os.path.join("tmp", "train_photos.csv"))[["photo_id"]]
    df = df[
        df["photo_id"].apply(
            lambda x: os.path.exists(os.path.join(working_dir, f"{x}.webp"))
        )
    ].reset_index(drop=True)
    df[["width", "height"]] = pd.DataFrame(
        df["photo_id"]
        .apply(lambda x: imagesize.get(os.path.join(working_dir, f"{x}.webp")))
        .tolist(),
        index=df.index,
    )
    df["width"] += 1
    df["height"] += 1

    dataset = ImagePatchDataset(
        df,
        working_dir,
        patch_size=256,
        transform=Compose([ToTensor(), RandomHorizontalFlip(), RandomVerticalFlip()]),
    )

    train_len = int(len(dataset) * 0.85)

    train_dataset, val_dataset = random_split(
        dataset, [train_len, len(dataset) - train_len]
    )
    val_dataset.dataset.cache_enabled = True

    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8
    )

    model = SRCNN().to(device)
    preprocessor = PreprocessingModule().to(device)
    import torch.optim as optim

    optimizer = optim.SGD(
        [
            {"params": model.conv1.parameters(), "lr": 1e-2},
            {"params": model.conv2.parameters(), "lr": 1e-2},
            {"params": model.conv3.parameters(), "lr": 1e-3},
        ],
        momentum=0.9,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )
    criterion = torch.nn.MSELoss().to(device)
    early_stop = EarlyStop()

    if start_epoch != 1:
        checkpoint = torch.load(
            os.path.join("tmp", "models", f"model_{start_epoch - 1}.tar")
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        early_stop.load_state_dict(checkpoint["early_stop"])
        print(
            f"Loaded checkpoint.\nEpoch {checkpoint['epoch']}: Train Loss = {checkpoint['train_loss']}, Val Loss: {checkpoint['val_loss']}"
        )

    for epoch in range(start_epoch, 101):
        train_loss = train_epoch(
            model, preprocessor, train_dataloader, optimizer, criterion, epoch
        )
        val_loss = val_epoch(model, preprocessor, val_dataloader, criterion)
        lr_scheduler.step(val_loss)
        stop_now = early_stop(val_loss)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "early_stop": early_stop.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
                "train_loss": train_loss,
            },
            os.path.join("tmp", "models", f"model2_{epoch}.tar"),
        )
        print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}")
        if stop_now:
            print("Stopping early.")
            break
