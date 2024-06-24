This script (ECGMULTI.ipynb) demonstrates how to classify ECG signals using a Convolutional Neural Network (CNN) in TensorFlow and Keras, designed to work with the MIT-BIH Supraventricular Arrhythmia Database. The script includes preprocessing steps such as normalization, label encoding, and dataset balancing using SMOTE, followed by downsampling. The data is split into training and test sets, standardized, and reshaped to fit the CNN model. A multi-layer CNN is built and trained to ensure robust evaluation. The model is trained with a validation split and batch size of 32 for 25 epochs, achieving high accuracy on both training and testing data. Update the file paths and configurations as needed for your dataset and computing environment.

This script (gui.py) provides a PyQt-based graphical user interface for classifying ECG signals using a pre-trained Convolutional Neural Network (CNN) model. The GUI allows users to upload a pre-trained model and a new ECG dataset for prediction. It includes preprocessing steps such as normalization, label encoding, and dataset balancing using SMOTE, followed by downsampling to ensure consistency with the training process. Users can make predictions on the new dataset and view the classification report, which includes precision, recall, f1-score, and support for each class. This tool is designed to simplify the process of evaluating ECG data with a trained CNN model. Update the file paths and configurations as needed for your dataset and computing environment.

