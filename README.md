# Chest X-Ray Pneumonia Detection using CNN

This project implements a convolutional neural network (CNN) to classify chest X-ray images as **NORMAL** or **PNEUMONIA**.

---

## Dataset

The dataset used is the Chest X-Ray Pneumonia dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Download the dataset and place the folder as:

Desktop/chest_xray/chest_xray/

---

## Requirements

Install dependencies using:

pip install -r requirements.txt

---

## Running the Code

Run the following command:

python train_cnn.py

The script will:

- Train the CNN
- Evaluate on validation and test sets
- Display predicted chest X-ray images with labels and probabilities

---

## Results

The model achieves:

- High pneumonia recall
- Approximately 80% test accuracy
- AUC near 0.9

---

## Author

Nathan Hu
