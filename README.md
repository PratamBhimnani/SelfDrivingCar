# Self-Driving Car Project

This project implements a self-driving car using behavioral cloning techniques. The main components of the project include the model training in a Jupyter notebook and a Python script to run the trained model and control the car.

## Project Structure

- **Behavioural_cloning.ipynb:** Jupyter notebook containing the implementation of the behavioral cloning model.
- **Drive.py:** Python script to run the trained model and control the car in a simulated environment.

## Files Description

### 1. Behavioural_cloning.ipynb
This Jupyter notebook includes the following steps:
- **Data Collection:** Using a simulator to collect driving data.
- **Data Preprocessing:** Processing the images and steering angles for training.
- **Model Architecture:** Building a convolutional neural network (CNN) to predict steering angles from images.
- **Model Training:** Training the CNN with the processed data.
- **Model Evaluation:** Evaluating the model performance on a validation set.
- **Model Saving:** Saving the trained model for later use in the driving script.

### 2. Drive.py
This Python script uses the trained model to drive the car in a simulated environment. It includes:
- **Model Loading:** Loading the pre-trained model.
- **Real-Time Prediction:** Capturing images from the simulator, preprocessing them, and predicting the steering angle using the model.
- **Control Commands:** Sending control commands (steering angle, throttle) to the simulator.

## Requirements

To run this project, you need the following dependencies:
- Python 3.x
- Keras
- TensorFlow
- NumPy
- OpenCV
- Flask (for the driving script)
- A self-driving car simulator (e.g., Udacity's Self-Driving Car Simulator)

You can install the required Python packages using:
```sh
pip install -r requirements.txt
```

## Running the Project

### 1. Train the Model
Open the `Behavioural_cloning.ipynb` notebook and run all the cells to train the model. The trained model will be saved as `model.h5`.

### 2. Run the Driving Script
Run the driving script using the following command:
```sh
python Drive.py model.h5
```

Make sure the simulator is running and configured to the correct mode (e.g., autonomous mode) to receive the control commands from the script.

## Acknowledgements

This project was inspired by the behavioral cloning project from the Udacity Self-Driving Car Nanodegree program.