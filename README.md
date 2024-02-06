# Pose Classification Project

<p float="left">
  <img src="https://i.imgur.com/r5skk8r.png" width="400" height="450"/>
  <img src="https://i.imgur.com/7ODrwxS.png" width="400" height="450"/> 
</p>

## Introduction
This project uses MoveNet for real-time pose estimation and a custom neural network model for pose classification, specifically for squat classification. It detects and counts squat reps performed by the user. The application is built with Streamlit and OpenCV for easy use and visualization.

## Installation
To set up the project, follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.

## Usage
Run the Streamlit app with:
- type `streamlit run streamlit_app.py` on your terminal.
- Place yourself in the camera's view and start performing squats. The application will display real-time pose estimation and count your squat reps.

## Training Your Own Model
If you're interested in training your own classification model, you can refer to this tutorial for guidance on how to get started:
- [Pose Classification Tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/pose_classification.ipynb)

This tutorial provides step-by-step instructions on how to train a pose classification model using TensorFlow Lite.


## License
MIT
