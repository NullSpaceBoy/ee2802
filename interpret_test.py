import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

avg_psnr = []
avg_ssim = []
avg_psnr_pre = []
avg_ssim_pre = []

avg_psnr_2 = []
avg_ssim_2 = []

for i in range(1, 21):
    data = pd.read_csv(os.path.join("tmp", "results", f"srcnn_{i}.csv"))
    data = data[data["psnr_pre"] != float("inf")].reset_index(drop=True)
    avg_psnr.append(data["psnr"].mean())
    avg_ssim.append(data["ssim"].mean())
    avg_psnr_pre.append(data["psnr_pre"].mean())
    avg_ssim_pre.append(data["ssim_pre"].mean())

for i in range(1, 6):
    data = pd.read_csv(os.path.join("tmp", "results", f"srcnn_{i}.csv"))
    avg_psnr_2.append(data["psnr"].mean())
    avg_ssim_2.append(data["ssim"].mean())

epochs = np.arange(1, 21)
epochs2 = np.arange(1, 6)

plt.plot(epochs, avg_psnr, label="500 images")
plt.plot(epochs2, avg_psnr_2, label="750 images")
plt.plot(epochs, avg_psnr_pre, label="Bicubic")
plt.legend()
plt.xlabel("Epoch")
plt.xticks(epochs)
plt.ylabel("PSNR (dB)")
plt.title("PSNR")
plt.figure()
plt.plot(epochs, avg_ssim, label="500 images")
plt.plot(epochs2, avg_ssim_2, label="750 images")
plt.plot(epochs, avg_ssim_pre, label="Bicubic")
plt.legend()
plt.xlabel("Epoch")
plt.xticks(epochs)
plt.ylabel("SSIM")
plt.title("SSIM")
plt.show()
