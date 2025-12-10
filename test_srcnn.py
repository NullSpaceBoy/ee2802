import os
import sys

import imagesize
import pandas as pd
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from img_patch_dataset import ImagePatchDataset
from srcnn import SRCNN
from srcnn_preprocessor import PreprocessingModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, preprocessor, loader, criteria):
    model.eval()
    pre_results = [[] for _ in criteria]
    results = [[] for _ in criteria]
    with torch.no_grad():
        for idx, hr_batch in tqdm(loader, smoothing=0):
            hr_batch = hr_batch.to(device)
            sr_batch = preprocessor(hr_batch)
            tmp = [criterion(sr_batch, hr_batch) for criterion in criteria]
            for i in range(len(criteria)):
                pre_results[i].extend(
                    (j.item(), value.item()) for j, value in zip(idx, tmp[i])
                )
            sr_batch = model(sr_batch)
            tmp = [criterion(sr_batch, hr_batch) for criterion in criteria]
            for i in range(len(criteria)):
                results[i].extend(
                    (j.item(), value.item()) for j, value in zip(idx, tmp[i])
                )
    return pre_results, results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_srcnn.py <working_dir>")
        sys.exit(1)

    working_dir = os.path.join(sys.argv[1], "test")

    df = pd.read_csv(os.path.join("tmp", "test_photos.csv"))[["photo_id"]]
    df[["width", "height"]] = pd.DataFrame(
        df["photo_id"]
        .apply(lambda x: imagesize.get(os.path.join(working_dir, f"{x}.webp")))
        .tolist(),
        index=df.index,
    )
    df["width"] += 1
    df["height"] += 1

    dataset = ImagePatchDataset(
        df, working_dir, patch_size=256, cache_enabled=True, idx=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
    )

    model = SRCNN().to(device)
    preprocessor = PreprocessingModule().to(device)
    ssim = StructuralSimilarityIndexMeasure(reduction=None).to(device)
    psnr = PeakSignalNoiseRatio(reduction=None, dim=(1, 2, 3), data_range=1.0).to(
        device
    )

    for epoch in range(1, 21):
        model.load_state_dict(
            torch.load(os.path.join("tmp", "models", f"model_{epoch}.tar"))["model"]
        )

        (ssim_pre_results, psnr_post_results), (ssim_results, psnr_results) = test(
            model, preprocessor, dataloader, [ssim, psnr]
        )

        ssim_pre_df = pd.DataFrame(ssim_pre_results, columns=["idx", "ssim_pre"])
        psnr_pre_df = pd.DataFrame(psnr_post_results, columns=["idx", "psnr_pre"])
        ssim_df = pd.DataFrame(ssim_results, columns=["idx", "ssim"])
        psnr_df = pd.DataFrame(psnr_results, columns=["idx", "psnr"])
        df = pd.merge(ssim_df, psnr_df, on="idx")
        df_pre = pd.merge(ssim_pre_df, psnr_pre_df, on="idx")
        df = pd.merge(df, df_pre, on="idx")
        os.makedirs(os.path.join("tmp", "results"), exist_ok=True)
        df.to_csv(os.path.join("tmp", "results", f"srcnn_{epoch}.csv"), index=False)

    for epoch in range(1, 6):
        model.load_state_dict(
            torch.load(os.path.join("tmp", "models", f"model2_{epoch}.tar"))["model"]
        )

        _, [ssim_results, psnr_results] = test(
            model, preprocessor, dataloader, [ssim, psnr]
        )
        print(ssim_results, psnr_results)
        ssim_df = pd.DataFrame(ssim_results, columns=["idx", "ssim"])
        psnr_df = pd.DataFrame(psnr_results, columns=["idx", "psnr"])
        df = pd.merge(ssim_df, psnr_df, on="idx")
        os.makedirs(os.path.join("tmp", "results"), exist_ok=True)
        df.to_csv(os.path.join("tmp", "results", f"srcnn2_{epoch}.csv"), index=False)
