#!/bin/bash

# Setup script for hand tracking functionality in SAMGA backend
echo "Setting up hand tracking for SAMGA backend..."

# Install required Python packages
pip install opencv-python==4.8.0.74
pip install numpy==1.24.3

# Update the requirements.txt file
if ! grep -q "opencv-python" requirements.txt; then
  echo "Adding OpenCV to requirements.txt"
  echo "opencv-python==4.8.0.74" >> requirements.txt
fi

if ! grep -q "numpy" requirements.txt; then
  echo "Adding numpy to requirements.txt"
  echo "numpy==1.24.3" >> requirements.txt
fi

echo "Setup complete! Make sure to restart the backend server."
echo "To test the hand tracking, navigate to the ping pong game in the frontend app." 