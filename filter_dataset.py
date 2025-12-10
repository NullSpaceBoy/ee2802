import os

import pandas as pd

# Read the TSV file containing photos and their URLs.
# ASSUMPTION: Path of the TSV file is "../unsplash-lite/photos.csv000"
df = pd.read_csv("../unsplash-lite/photos.csv000", sep="\t", header=0)[
    ["photo_id", "photo_image_url"]
]

# Some links are invalid or no longer work. Only keep those that have the correct domain.
df = df[df["photo_image_url"].str.startswith("https://images.unsplash.com/")]

# Only consider the first 850 images (750 for training + 100 for testing)
df = df[:850].reset_index(drop=True)

train_df = df[:750]
test_df = df[750:]

try:
    os.mkdir("tmp")
except FileExistsError:
    pass

# Results are saved to tmp/train_photos.csv and tmp/test_photos.csv.
train_df.to_csv(os.path.join("tmp", "train_photos.csv"), index=False, mode="w+")
test_df.to_csv(os.path.join("tmp", "test_photos.csv"), index=False, mode="w+")
