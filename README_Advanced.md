# 🌍 Earthquake Classification Using CNN

## 📌 Overview
This project leverages Convolutional Neural Networks (CNNs) to classify seismic events as either **earthquakes** or **background noise** using grayscale spectrogram images. By processing waveform data from the Stanford Earthquake Dataset (STEAD), this work enhances seismic signal understanding to support early warning systems and seismic analysis.

## 🚀 Features
🔹 Classifies earthquake vs. non-earthquake signals using deep learning.  
🔹 Uses spectrogram images generated from raw waveform data.  
🔹 Built with TensorFlow & Keras for efficient CNN training.  
🔹 Supports dynamic dataset sizing (full or sampled).  
🔹 Evaluates performance using accuracy, precision, recall, and confusion matrices.  
🔹 Implements parallel processing for efficient image generation.

## 📂 Dataset
The project uses the **Stanford Earthquake Dataset (STEAD)**, which includes:

- 1.2 million seismic waveform samples in `.csv` and `.hdf5` format.
- Categories: `earthquake_local` and `noise`.
- Spectrogram images generated from the Z-channel of waveforms.

## 🏗 Model Architecture
The CNN model includes:

- **Convolutional Layers**: For feature extraction from spectrograms.  
- **MaxPooling Layers**: For downsampling and reducing overfitting.  
- **Dropout Layers**: Dropout rate = 0.25 for generalization.  
- **Fully Connected Layers**: To capture higher-level abstractions.  
- **Output Layer**: Softmax output for binary classification.

## 🛠 Tech Stack
- Python 🐍  
- TensorFlow & Keras 🔥  
- NumPy & Pandas 📊  
- Matplotlib & Seaborn 📉  
- OpenCV (for image processing)  
- HDF5 (for waveform storage)  
- Joblib (for parallel processing)

## 📊 Model Performance
- Test Accuracy: ~98%  
- Precision & Recall: Calculated per run  
- Evaluation Metrics:  
  - Confusion Matrix  
  - Accuracy/Loss plots  

The model shows strong performance distinguishing between noise and earthquake spectrograms.

## 🔮 Future Enhancements
🔹 Extend classification to include earthquake types (e.g., local vs. regional).  
🔹 Integrate real-time streaming data classification.  
🔹 Use transfer learning to improve performance on unseen datasets.  
🔹 Add more diverse data from different seismic networks.

## 📜 Research Paper
This project is part of a larger research initiative aimed at **applying deep learning to seismic signal analysis**, targeting submission to Scopus/SCI-indexed journals in Earth Sciences and AI.

## 📖 References
- Amirul Hoque et al. – Earthquake Magnitude Prediction Using Machine Learning (IEEE, 2024)  
- Michael Brown & Emily Davis – AI in Predicting Earthquakes: Challenges & Innovations (IEEE Access, 2020)  
- Satoshi Nakamura & Yuki Tanaka – Improving Earthquake Ground Motion Predictions with AI (Natural Hazards, 2024)

## 🤝 Contributing
We welcome contributions! Feel free to:

⭐ Star this repository if you find it useful.  
🐛 Open an issue for bug reports or suggestions.  
🛠 Submit pull requests with improvements.

## 📬 Contact
For questions or collaborations, connect via [LinkedIn](https://www.linkedin.com) or raise an issue on GitHub.

---

🚀 **Bringing AI to Earthquake Prediction for a Safer Future!** 🌍
