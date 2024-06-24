import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

class ECGApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ECG Classification")
        self.setGeometry(100, 100, 800, 600)

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = None

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.load_model_button = QPushButton("Load Pre-trained Model")
        self.load_model_button.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_button)

        self.load_data_button = QPushButton("Load Dataset")
        self.load_data_button.clicked.connect(self.load_dataset)
        layout.addWidget(self.load_data_button)

        self.predict_button = QPushButton("Make Predictions")
        self.predict_button.clicked.connect(self.make_predictions)
        layout.addWidget(self.predict_button)

        self.result_label = QLabel("Results will be displayed here.")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "H5 Files (*.h5);;All Files (*)", options=options)
        if file_name:
            self.model = load_model(file_name)
            QMessageBox.information(self, "Model Loaded", f"Model loaded from {file_name}")

    def load_dataset(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.data = pd.read_csv(file_name)
            if 'record' in self.data.columns:
                self.data = self.data.drop(columns=['record'])
            self.data = self.preprocess_data(self.data)
            QMessageBox.information(self, "Dataset Loaded", f"Dataset loaded from {file_name}")

    def preprocess_data(self, data):
        

        # Encode the class labels
        if 'type' in data.columns:
            self.label_encoder = LabelEncoder()
            data['type'] = self.label_encoder.fit_transform(data['type'])
            self.label_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
            y = data['type'].values
            data = data.drop(columns=['type'])
        else:
            raise ValueError("The 'type' column is missing from the dataset")

        # Normalize the input features
        X = data.values
        X_scaled = self.scaler.fit_transform(X)

        # Apply SMOTE to balance the dataset
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Downsample to 30 samples per class
        resampled_df = pd.DataFrame(X_resampled)
        resampled_df['type'] = y_resampled
        downsampled_df = resampled_df.groupby('type').apply(lambda x: x.sample(30, replace=True)).reset_index(drop=True)

        # Separate features and target variable
        X_downsampled = downsampled_df.drop(columns=['type']).values
        y_downsampled = downsampled_df['type'].values

        # Convert the labels to categorical (one-hot encoding)
        y_downsampled_categorical = to_categorical(y_downsampled)

        return X_downsampled, y_downsampled_categorical

    def make_predictions(self):
        if self.model is None:
            QMessageBox.warning(self, "Error", "Please load a pre-trained model first.")
            return
        if not hasattr(self, 'data'):
            QMessageBox.warning(self, "Error", "Please load a dataset first.")
            return

        X, y_true = self.data
        X = X.reshape(X.shape[0], X.shape[1], 1)
        predictions = self.model.predict(X)
        y_pred_classes = np.argmax(predictions, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)

        report = classification_report(y_true_classes, y_pred_classes, target_names=self.label_encoder.classes_)
        self.result_label.setText(report)
        QMessageBox.information(self, "Classification Report", report)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ECGApp()
    window.show()
    sys.exit(app.exec_())
