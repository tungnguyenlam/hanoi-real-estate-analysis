#!/bin/bash

# filepath: /Users/tungnguyen/Library/CloudStorage/GoogleDrive-nguyenlamtungthptltt@gmail.com/My Drive/Projects-Large/Midterm-Into2DS/hanoi-real-estate-analysis/set_up_dataset.sh

# Check if the current working directory is hanoi-real-estate-analysis
if [[ $(basename "$PWD") != "hanoi-real-estate-analysis" ]]; then
  echo "Please navigate to the 'hanoi-real-estate-analysis' folder and run this script."
  exit 1
fi

# Create the data/ folder if it doesn't exist
if [[ ! -d "data" ]]; then
  echo "Creating data/ folder..."
  mkdir data
fi

# Download the file from Google Drive
FILE_ID="1vuWugmic5XFwGcZlkuyvO9SsRD73F-lY"
DEST_FILE="data/dataset.zip"

echo "Downloading dataset from Google Drive..."
curl -L -o "$DEST_FILE" "https://drive.google.com/uc?id=${FILE_ID}&export=download"

# Check if the download was successful
if [[ $? -ne 0 ]]; then
  echo "Failed to download the dataset. Please check the link or your internet connection."
  exit 1
fi

# Unzip the downloaded file into the data/ folder
echo "Unzipping the dataset..."
unzip -o "$DEST_FILE" -d data/

# Check if the unzip was successful
if [[ $? -ne 0 ]]; then
  echo "Failed to unzip the dataset. Please check the downloaded file."
  exit 1
fi

# Remove the zip file after extraction
echo "Cleaning up..."
rm "$DEST_FILE"

echo "Dataset setup completed successfully!"