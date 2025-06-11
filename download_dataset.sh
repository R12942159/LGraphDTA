#!/bin/bash

echo "Downloading LGraphDTA_dataset.zip..."
gdown --id 1c1s9PrWr_FW_D0k-SF-goDx5p1BUNaXl -O LGraphDTA_dataset.zip

echo "Unzipping LGraphDTA_dataset.zip..."
unzip -o LGraphDTA_dataset.zip

echo "Cleaning up zip file..."
rm LGraphDTA_dataset.zip

echo "Dataset download and extraction complete."