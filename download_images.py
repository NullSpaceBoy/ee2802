import asyncio
import os
import sys

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

if len(sys.argv) != 2:
    print("Usage: uv run download_images.py <download_dir>")
    sys.exit(1)

download_dir = sys.argv[1]


async def download_file(session, photo_id, url, download_dir):
    """Download a single file asynchronously"""
    filename = os.path.join(download_dir, f"{photo_id}.webp")
    try:
        async with session.get(f"{url}?fm=webp&lossless=true") as response:
            if response.status == 200:
                content = await response.read()
                with open(filename, "wb") as f:
                    f.write(content)
                    return photo_id, "success"
            else:
                return photo_id, "failed"
    except Exception:
        return photo_id, "error"


async def download_all(df, download_dir):
    """Download all URLs from DataFrame concurrently"""
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)

    # Extract URLs from DataFrame
    urls = df["photo_image_url"].tolist()
    photo_ids = df["photo_id"].tolist()

    # Create aiohttp session and download concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_file(session, photo_id, url, download_dir)
            for url, photo_id in zip(urls, photo_ids)
        ]
        results = await tqdm_asyncio.gather(*tasks)

        return pd.DataFrame(results, columns=["id", "status"])


train_df = pd.read_csv(os.path.join("tmp", "train_photos.csv"))[
    ["photo_id", "photo_image_url"]
]

test_df = pd.read_csv(os.path.join("tmp", "test_photos.csv"))[
    ["photo_id", "photo_image_url"]
]

# Ignore images that are already present
files = []
try:
    files = os.listdir(os.path.join(download_dir, "train"))
except FileNotFoundError:
    os.makedirs(os.path.join(download_dir, "train"), exist_ok=True)
train_df = train_df[~train_df["photo_id"].apply(lambda x: f"{x}.webp" in files)]

try:
    files = os.listdir(os.path.join(download_dir, "test"))
except FileNotFoundError:
    files = []
    os.makedirs(os.path.join(download_dir, "test"), exist_ok=True)
test_df = test_df[~test_df["photo_id"].apply(lambda x: f"{x}.webp" in files)]


class EarlyStop:
    def __init__(self, curr, patience=5):
        self.prev = curr
        self.patience = patience
        self.counter = 0

    def __call__(self, curr):
        if curr != self.prev:
            self.counter = 0
            self.prev = curr
        else:
            self.counter += 1
        return self.counter >= self.patience


early_stop = EarlyStop(0)
while len(train_df) != 0 and not early_stop(len(train_df)):
    results = asyncio.run(download_all(train_df, os.path.join(download_dir, "train")))
    train_df = train_df[
        train_df["photo_id"].isin(results[results["status"] != "success"]["id"])
    ]

early_stop = EarlyStop(0)
while len(test_df) != 0 and not early_stop(len(test_df)):
    results = asyncio.run(download_all(test_df, os.path.join(download_dir, "test")))
    test_df = test_df[
        test_df["photo_id"].isin(results[results["status"] != "success"]["id"])
    ]

# Some images may not have been downloaded successfully. Remove them from the DataFrame and save the filtered dataframe.
df = pd.read_csv(os.path.join("tmp", "train_photos.csv"))
df = df[~df["photo_id"].isin(train_df["photo_id"])]
df.to_csv(os.path.join("tmp", "train_photos.csv"), index=False)

df = pd.read_csv(os.path.join("tmp", "test_photos.csv"))
df = df[~df["photo_id"].isin(test_df["photo_id"])]
df.to_csv(os.path.join("tmp", "test_photos.csv"), index=False)
