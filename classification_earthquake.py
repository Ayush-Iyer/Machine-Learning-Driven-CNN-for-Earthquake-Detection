
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import cv2

import tensorflow as tf
from tensorflow.keras import layers

keras = tf.keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime


############################# USER INPUT #############################

# set CNN data paths

# path to image dataset
dir = 'D:/Earthquake_2/Classification/Images/'

# path to image dataset
data_dir = 'D:/Earthquake_2/DATASET/'


# paths to csv and hdf5 (waveform/signal) files
noise_csv_path = data_dir+'chunk1/chunk1.csv'
noise_sig_path = data_dir+'chunk1/chunk1.hdf5'
eq1_csv_path = data_dir+'chunk2/chunk2.csv'
eq1_sig_path = data_dir+'chunk2/chunk2.hdf5'
# eq2_csv_path = data_dir+'data/chunk3/chunk3.csv'
# eq2_sig_path = data_dir+'data/chunk3/chunk3.hdf5'
# eq3_csv_path = data_dir+'data/chunk4/chunk4.csv'
# eq3_sig_path = data_dir+'data/chunk4/chunk4.hdf5'
# eq4_csv_path = data_dir+'data/chunk5/chunk5.csv'
# eq4_sig_path = data_dir+'data/chunk5/chunk5.hdf5'
# eq5_csv_path = data_dir+'data/chunk6/chunk6.csv'
# eq5_sig_path = data_dir+'data/chunk6/chunk6.hdf5'

# read the noise and earthquake csv files into separate dataframes:
earthquakes_1 = pd.read_csv(eq1_csv_path)
# earthquakes_2 = pd.read_csv(eq2_csv_path)
# earthquakes_3 = pd.read_csv(eq3_csv_path)
# earthquakes_4 = pd.read_csv(eq4_csv_path)
# earthquakes_5 = pd.read_csv(eq5_csv_path)
noise = pd.read_csv(noise_csv_path)

full_csv = pd.concat([earthquakes_1,noise])
# full_csv = pd.concat([earthquakes_1,earthquakes_2,earthquakes_3,earthquakes_4,earthquakes_5,noise])

# making lists of trace names for the earthquake sets
eq1_list = earthquakes_1['trace_name'].to_list()
# eq2_list = earthquakes_2['trace_name'].to_list()
# eq3_list = earthquakes_3['trace_name'].to_list()
# eq4_list = earthquakes_4['trace_name'].to_list()
# eq5_list = earthquakes_5['trace_name'].to_list()

# making a list of trace names for the noise set
noise_list = noise['trace_name'].to_list()


#######################################################################


class SeismicCNN:

    def __init__(self, model_type, choose_dataset_size, full_csv, dir):
        self.model_type = model_type
        self.choose_dataset_size = choose_dataset_size
        self.full_csv = full_csv
        self.dir = dir
        self.traces_array = []
        self.img_dataset = []
        self.labels = []
        self.imgs = []
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.model = []
        self.test_loss = []
        self.test_acc = []
        self.predicted_classes = []
        self.predicted_probs = []
        self.cm = []
        self.epochs = []
        self.history = []

        # Classification initialization
        if self.model_type == 'classification':
            print('Creating seismic trace list')
            for filename in os.listdir(dir):
                if filename.endswith('.png'):
                    self.traces_array.append(filename[0:-4])

            if choose_dataset_size == 'full':
                print('Selecting traces matching images in directory')
                self.img_dataset = self.full_csv.loc[self.full_csv['trace_name'].isin(self.traces_array)]
                self.labels = self.img_dataset['trace_category']
                self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0))
                print(f'The number of traces in the directory is {len(self.img_dataset)}')

                for i in range(0, len(self.img_dataset['trace_name'])):
                    img = cv2.imread(self.dir + '/' + self.img_dataset['trace_name'].iloc[i] + '.png', 0)
                    self.imgs.append(img)
                self.imgs = np.array(self.imgs)

            elif type(choose_dataset_size) == int:
                seismic_dataset = self.full_csv.loc[self.full_csv['trace_name'].isin(self.traces_array)]
                choose_seismic_dataset = np.random.choice(np.array(seismic_dataset['trace_name']), choose_dataset_size, replace=False)
                self.img_dataset = seismic_dataset.loc[seismic_dataset['trace_name'].isin(choose_seismic_dataset)]
                self.labels = self.img_dataset['trace_category']
                self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0))
                print(f'The number of traces in the directory is {len(self.img_dataset)}')

                for i in range(0, len(self.img_dataset['trace_name'])):
                    img = cv2.imread(self.dir + '/' + self.img_dataset['trace_name'].iloc[i] + '.png', 0)
                    self.imgs.append(img)
                self.imgs = np.array(self.imgs)
            else:
                print('Error: choose "full" for full dataset or provide an integer for random samples.')

    def train_test_split(self, test_size, random_state):
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(
            self.imgs, self.labels, random_state=random_state, test_size=test_size)

        print(f'The training images set size: {self.train_images.shape}')
        print(f'The testing images set size: {self.test_images.shape}')

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

        img_height = self.train_images.shape[1]
        img_width = self.train_images.shape[2]

        print('Resizing images')
        self.train_images = self.train_images.reshape(-1, img_height, img_width, 1)
        self.test_images = self.test_images.reshape(-1, img_height, img_width, 1)

    def classification_cnn(self, epochs):
        self.epochs = epochs
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=f'./saved_models/specs_{str(self.choose_dataset_size)}dataset_{self.model_type}_epochs{self.epochs}_{datetime.now().strftime("%Y%m%d")}', save_freq='epoch')
        ]

        print('Building CNN model')
        model = keras.Sequential()
        model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
        model.add(layers.MaxPool2D(2, 2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten(input_shape=(self.imgs.shape[1], self.imgs.shape[2])))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate=1e-6)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics='accuracy')
        self.history = model.fit(self.train_images, self.train_labels, batch_size=64, epochs=epochs, callbacks=callbacks, validation_split=0.2)

        print(model.summary())
        self.model = model

    def evaluate_classification_model(self):
        print('Evaluating model on test dataset')
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=1)
        print(f"\nTest data accuracy: {100 * self.test_acc:.2f}%")

        self.predicted_classes = np.argmax(self.model.predict(self.test_images), axis=-1)
        self.predicted_probs = self.model.predict(self.test_images)

        print('Building confusion matrix')
        self.cm = confusion_matrix(self.test_labels, self.predicted_classes)
        print(self.cm)
        accuracy = accuracy_score(self.test_labels, self.predicted_classes)
        precision = precision_score(self.test_labels, self.predicted_classes)
        recall = recall_score(self.test_labels, self.predicted_classes)
        print(f'Model accuracy: {accuracy}, precision: {precision}, recall: {recall}.')

        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['not earthquake', 'earthquake'])
        disp.plot(cmap='Blues', values_format='')
        plt.title(f'Classification CNN Results ({self.epochs} epochs)')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()

        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(self.history.history['accuracy'])
        ax.plot(self.history.history['val_accuracy'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['train', 'test'])
        plt.savefig('model_accuracy.png')
        plt.show()


# Example usage for classification CNN
model_cnn_c1 = SeismicCNN('classification', 8000, full_csv, dir)
model_cnn_c1.train_test_split(test_size=0.25, random_state=44)
model_cnn_c1.classification_cnn(10)
model_cnn_c1.evaluate_classification_model()
