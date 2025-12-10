import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if __name__ == "__main__":
    dir = os.path.join("tmp", "models")

    data = [
        torch.load(
            os.path.join(dir, f"model_{i}.tar"), map_location=torch.device("cpu")
        )
        for i in range(
            1, len(list(filter(lambda x: x.startswith("model_"), os.listdir(dir)))) + 1
        )
    ]
    data2 = [
        torch.load(
            os.path.join(dir, f"model2_{i}.tar"), map_location=torch.device("cpu")
        )
        for i in range(
            1, len(list(filter(lambda x: x.startswith("model2_"), os.listdir(dir)))) + 1
        )
    ]

    data = pd.DataFrame(data)[["epoch", "train_loss", "val_loss"]]
    data2 = pd.DataFrame(data2)[["epoch", "train_loss", "val_loss"]]

    plt.plot(data["epoch"], np.log10(data["train_loss"]), label="Training")
    plt.plot(data["epoch"], np.log10(data["val_loss"]), label="Validation")
    plt.legend()
    plt.ylabel("$\\log_{10}$MSE")
    plt.xlabel("Epoch")
    plt.xticks(data["epoch"])
    plt.title("Training and Validation Loss (500 images)")

    plt.figure()
    plt.plot(data2["epoch"], np.log10(data2["train_loss"]), label="Training")
    plt.plot(data2["epoch"], np.log10(data2["val_loss"]), label="Validation")
    plt.legend()
    plt.ylabel("$\\log_{10}$MSE")
    plt.xlabel("Epoch")
    plt.xticks(data2["epoch"])
    plt.title("Training and Validation Loss (750 images)")

    plt.figure()
    plt.plot(data["epoch"], np.log10(data["train_loss"]), label="500 images")
    plt.plot(data2["epoch"], np.log10(data2["train_loss"]), label="750 images")
    plt.legend()
    plt.ylabel("$\\log_{10}$MSE")
    plt.xlabel("Epoch")
    plt.xticks(data["epoch"] if len(data) > len(data2) else data2["epoch"])
    plt.title("Validation Loss Comparison")

    plt.show()
