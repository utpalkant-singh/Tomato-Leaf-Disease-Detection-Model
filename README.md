# Tomato-Leaf-Disease-Detection-Model
A ML model based on CNN for detecting tomato leaf disease.

## Tomato Leaf Disease Detection using CNN
This project aims to detect the diseases present in tomato plant leaves using Convolutional Neural Networks (CNNs). The dataset used in this project consists of images of healthy tomato leaves and leaves infected with three common diseases: early blight, late blight, and leaf mold.

## Dataset
The dataset used in this project was collected from various sources and contains a total of 10,000 images of tomato leaves. These images are divided into four categories: healthy, early blight, late blight, and leaf mold. The dataset was split into training and testing sets with a ratio of 70:15:15.

## Libraries Used
This project was implemented using Python programming language and the following libraries were used:

TensorFlow
Keras
NumPy
Matplotlib

## Model Architecture
The CNN model used in this project consists of four convolutional layers, each followed by a max pooling layer. The output from the last convolutional layer is flattened and fed into a dense layer with 128 neurons, which is then connected to the output layer with four neurons (one for each class).

## Training and Testing
The model was trained on the training set for 150 epochs with a batch size of 128. The model achieved an accuracy of 77.87% on the training set and 75.65% on the testing set.

## How to Use
To use this project, follow the steps below:

Clone the repository to your local machine
Install the required libraries (TensorFlow, Keras, NumPy, Matplotlib)
Run the Tomato_Disease_Classification.ipynb
## Credits
The dataset used in this project was collected from the following sources:

PlantVillage
Kaggle
## Conclusion
This project demonstrates the effectiveness of CNNs in detecting tomato leaf diseases. The trained model achieved a high accuracy on both the training and testing sets, indicating its potential for use in real-world applications.
