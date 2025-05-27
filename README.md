# Earthquake Classification Using Convolutional Neural Networks (CNN)

This project focuses on classifying seismic events (earthquakes vs. noise) using spectrogram images derived from waveform data. It uses Convolutional Neural Networks (CNNs) to distinguish between earthquake and non-earthquake events based on seismic signals.

## 📁 Project Structure

- `create_images.py`: Converts seismic waveform data from HDF5 files into spectrogram images.
- `classification_earthquake.py`: Defines and trains a CNN model using TensorFlow/Keras to classify the spectrogram images.
- `Images/`: Directory where the spectrogram PNG files are stored.
- `DATASET/`: Folder containing seismic waveform chunks in `.csv` and `.hdf5` formats.

## 🔧 Setup Instructions

### Prerequisites

Install the following packages:

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn h5py joblib tensorflow
```

### Folder Structure Example

```
D:/Earthquake_2/
├── DATASET/
│   ├── chunk1/
│   │   ├── chunk1.csv
│   │   └── chunk1.hdf5
│   ├── chunk2/
│   │   ├── chunk2.csv
│   │   └── chunk2.hdf5
├── Classification/
│   └── Images/
```

Update the `dir`, `data_dir`, and paths in both Python files accordingly to match your environment.

## 📷 Image Generation

Run `create_images.py` to generate spectrograms from the seismic waveforms:

```bash
python create_images.py
```

This will save `.png` images into the `Images/` directory.

## 🧠 CNN Model Training

Run `classification_earthquake.py` to:

1. Load the image data.
2. Preprocess it for training.
3. Define and train a CNN.
4. Evaluate the model with metrics and visualizations (accuracy, confusion matrix).

```bash
python classification_earthquake.py
```

You can customize:
- Dataset size (full or integer sample size)
- Number of training epochs
- Test/train split ratio

## 📊 Output

The script will generate:
- `model_accuracy.png`: Plot of training/validation accuracy per epoch.
- `confusion_matrix.png`: Confusion matrix visualizing true/false positives/negatives.

## 🧪 Example Code Usage

```python
model = SeismicCNN('classification', 8000, full_csv, dir)
model.train_test_split(test_size=0.25, random_state=44)
model.classification_cnn(epochs=10)
model.evaluate_classification_model()
```

## 📌 Notes

- Earthquake data is expected to have a label `earthquake_local`; all others are treated as noise.
- Images are grayscale 2D spectrograms created from the Z-channel of the waveform.
- Ensure that only traces with corresponding `.png` files are included during model training.

## 📄 License

This project is for educational and research purposes.
