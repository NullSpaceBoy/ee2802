import os
import sys

import imagesize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

if len(sys.argv) != 2:
    print("Usage: uv run dataset_analysis.py <download_dir>")
    sys.exit(1)

working_dir = os.path.join(sys.argv[1], "train")

df = pd.read_csv(os.path.join("tmp", "train_photos.csv"))[["photo_id"]]

df[["width", "height"]] = pd.DataFrame(
    df["photo_id"]
    .apply(lambda x: imagesize.get(os.path.join(working_dir, f"{x}.webp")))
    .tolist(),
    index=df.index,
)
df["width"] += 1
df["height"] += 1

# Set the graph axis limits to ignore extreme values.
w_min, w_max = df["width"].quantile([0.01, 0.99])
h_min, h_max = df["height"].quantile([0.01, 0.99])

g = seaborn.jointplot(
    data=df,
    x="width",
    y="height",
    kind="kde",
    fill=True,
    cmap="viridis",
    thresh=0.05,
    xlim=(w_min, w_max),
    ylim=(h_min, h_max),
    levels=50,
    bw_adjust=0.25,
)
g.figure.suptitle("Density Map of Photo Dimensions")
g.set_axis_labels("Photo Width", "Photo Height")
plt.tight_layout()
plt.show()
