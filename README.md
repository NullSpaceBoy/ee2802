## Prerequisites
1. Ensure that the [Unsplash Lite](https://github.com/unsplash/datasets) dataset is downloaded and saved as `../unsplash-lite`.
2. Ensure that `uv` is installed.

## Preparing the Dataset
1. **Filter the dataset to use only a small subset of images**: Create a new list of images that will be stored in `tmp/train_photos.csv` and `tmp/test_photos.csv`.
   ```bash
   uv run filter_dataset.py
   ```
2. **Download the images**: Download all the images listed in `tmp/train_photos.csv` and `tmp/test_photos.csv` to `<working_directory>/train` and `<working_directory>/test` respectively and filter out any images that failed to download.
   ```bash
   uv run download_images.py <working_directory>
   ```
## Dataset Analysis
```bash
uv run dataset_analysis.py <working_directory>
```
generates a plot of the distribution of images by dimensions.

## Training
```bash
uv run train_srcnn.py <working_directory> <start_epoch>
```
`start_epoch` is 1 by default. This trains and validates the model for upto 100 epochs. For each epoch, after validation, the training and validation loss along with the states of the model, optimizer and scheduler are saved to `tmp/models/model_<epoch>.tar`. If `start_epoch` is not 1, the model from the previous epoch will be loaded by accessing `tmp/models/model_<start_epoch-1>.tar`.

## Evaluation
```bash
uv run test_srcnn.py <working_directory>
```
This computes the average PSNR and SSIM over the test dataset at all epochs for which a `tmp/models/model_<epoch>.tar` exists.

## Plotting Loss vs Epochs
```bash
uv run interpret_training.py
```
This reads the losses from `tmp/models/model_<epoch>.tar`.

## Plotting PSNR and SSIM vs Epochs
```bash
uv run interpret_test.py <working_directory>
```
This reads the PSNR and SSIM from `tmp/models/model_<epoch>.tar` and plots them against the epoch number.

## Visualizing the Filters
```bash
uv run get_filters.py <epoch>
```
This loads the model from `tmp/models/model_<epoch>.tar` and renders the filters as a grid in a grayscale image.
