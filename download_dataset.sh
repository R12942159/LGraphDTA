#!/bin/bash

echo "Downloading LGraphDTA_dataset.zip..."
gdown --id 1uMWAiq9sxZIQ9Ro0M5bxIi1Zbgex6izw -O LGraphDTA_dataset.zip

echo "Unzipping LGraphDTA_dataset.zip..."
unzip -o LGraphDTA_dataset.zip

echo "Cleaning up zip file..."
rm LGraphDTA_dataset.zip

echo "Dataset download and extraction complete."