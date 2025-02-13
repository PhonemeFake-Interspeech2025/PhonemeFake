#!/bin/bash

FILE_ID="1vkx5wI978aClfsDcjUuHK8gyrFGzAlwJ"
FILE_NAME="conf-3-linear.pth"

# Step 1: Fetch cookie
curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null

# Step 2: Use confirmation token to download file
curl -Lb ./cookie.txt "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' ./cookie.txt)&id=${FILE_ID}" -o ${FILE_NAME}

echo "Downloaded ${FILE_NAME}"
