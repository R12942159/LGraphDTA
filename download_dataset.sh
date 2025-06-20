#!/bin/bash

echo "Downloading LGraphDTA_dataset.zip..."
gdown --id 1M3OejPpJE9uO0bcFjqDjs4cJ-QYaYvs4 -O LGraphDTA_dataset.zip

echo "Unzipping LGraphDTA_dataset.zip..."
unzip -o LGraphDTA_dataset.zip

echo "Cleaning up dataset zip..."
rm LGraphDTA_dataset.zip

echo "Downloading ckpt-LGraphDTA.zip..."
gdown --id 1ks1-eql35gHi-7kKksvwJAXb0w5a5vO_ -O ckpt-LGraphDTA.zip

echo "Unzipping ckpt-LGraphDTA.zip..."
unzip -o ckpt-LGraphDTA.zip

echo "Cleaning up ckpt zip..."
rm ckpt-LGraphDTA.zip

echo "All downloads and extractions completed successfully."