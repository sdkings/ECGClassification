import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ECGApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.filepath = None
        self.df = None
        self.X_balanced = None
        self.y_balanced = None
        self.model = None
        self.X_test = None
        self.y_test_categorical = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("ECG Classification")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.upload_button = QPushButton("Upload Dataset", self)
        self.upload_button.clicked.connect(self.upload_dataset)
        layout.addWidget(self.upload_button)

        self.preprocess_button = QPushButton("Preprocess Dataset", self)
        self.preprocess_button.clicked.connect(self.preprocess_dataset)
        layout.addWidget(self.preprocess_button)

        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.results_button = QPushButton("Show Results", self)
        self.results_button.clicked.connect(self.show_results)
        layout.addWidget(self.results_button)

        self.text_edit = QTextEdit(self)
        layout.addWidget(self.text_edit)

        self.canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_dataset(self):
        options = QFileDialog.Options()
        self.filepath, _ = QFileDialog.getOpenFileName(self, "Upload Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if self.filepath:
            self.df = pd.read_csv(self.filepath)
            self.text_edit.append(f"Dataset loaded with shape: {self.df.shape}")
            self.text_edit.append(str(self.df.info()))
            self.plot_class_distribution(self.df, title="Class Distribution Before Preprocessing")

    def preprocess_dataset(self):
        if self.df is None:
            self.text_edit.append("No dataset uploaded")
            return

        # Drop the 'record' column
        if 'record' in self.df.columns:
            self.df = self.df.drop(columns=['record'])

        # Encode the class labels
        label_encoder = LabelEncoder()
        self.df['type'] = label_encoder.fit_transform(self.df['type'])

        # Normalize the input features
        scaler = StandardScaler()
        X = self.df.drop(columns=['type']).values
        X_scaled = scaler.fit_transform(X)

        # Extract the target variable
        y = self.df['type'].values

        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Combine resampled data into a DataFrame for easy handling
        resampled_df = pd.DataFrame(X_resampled)
        resampled_df['type'] = y_resampled

        # Take exactly 20,000 samples per class
        desired_samples_per_class = 20000
        balanced_df = resampled_df.groupby('type').apply(lambda x: x.sample(n=desired_samples_per_class, random_state=42)).reset_index(drop=True)

        # Separate features and target
        self.X_balanced = balanced_df.drop(columns=['type']).values
        self.y_balanced = balanced_df['type'].values

        self.text_edit.append("Dataset preprocessing complete")
        self.plot_class_distribution(balanced_df, title="Class Distribution After Preprocessing")

    def create_model(self):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(self.X_balanced.shape[1], 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(np.unique(self.y_balanced)), activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        if self.X_balanced is None or self.y_balanced is None:
            self.text_edit.append("No dataset preprocessed")
            return

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_balanced, self.y_balanced, test_size=0.2, random_state=42)

        # Reshape the input data to fit the CNN model
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Convert the labels to categorical (one-hot encoding)
        y_train_categorical = to_categorical(y_train)
        y_test_categorical = to_categorical(y_test)

        # Create and train the model
        self.model = self.create_model()
        self.model.fit(X_train, y_train_categorical, validation_data=(X_test, y_test_categorical), epochs=25, batch_size=32)

        # Save the test data for later use in results
        self.X_test = X_test
        self.y_test_categorical = y_test_categorical

        # Save the model
        self.model.save('ecg_model.h5')

        self.text_edit.append("Model training complete. Model saved as 'ecg_model.h5'")

    def show_results(self):
        if self.model is None:
            self.text_edit.append("No model trained")
            return

        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test_categorical)
        self.text_edit.append(f'Test Accuracy: {test_accuracy:.4f}')

        # Generate predictions and classification report
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = y_pred.argmax(axis=1)
        y_test_classes = self.y_test_categorical.argmax(axis=1)
        report = classification_report(y_test_classes, y_pred_classes, output_dict=True)
        self.text_edit.append("Classification Report:")
        self.text_edit.append(str(report))

        # Display classification report in the GUI
        report_text = classification_report(y_test_classes, y_pred_classes)
        self.text_edit.append(report_text)

    def plot_class_distribution(self, df, title):
        class_counts = df['type'].value_counts()
        labels = class_counts.index
        sizes = class_counts.values

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10}, colors=plt.cm.Paired.colors)

        ax.legend(wedges, labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=10, weight="bold")
        plt.title(title, fontsize=14)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Embed the plot into PyQt5
        self.canvas.figure.clf()
        self.canvas.figure = fig
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ECGApp()
    ex.show()
    sys.exit(app.exec_())
