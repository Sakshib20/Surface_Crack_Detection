# ⚙️ Surface Crack Detection: CNN-Based Industrial Inspection

# 📌 Project Overview

This project is an industrial-grade Computer Vision application designed for automated structural health monitoring. By implementing a Convolutional Neural Network (CNN), the system analyzes surface textures to accurately detect and classify structural cracks into "Crack" and "No Crack" categories. The application automates the end-to-end pipeline, including recursive dataset splitting, image augmentation, deep learning model training, and real-time inference.

# 🛠 Tech Stack & Environment

Language: Python 3.x

Deep Learning Framework: TensorFlow 2.x, Keras (Sequential API)

Computer Vision: ImageDataGenerator, NumPy

Evaluation Metrics: Scikit-learn (Confusion Matrix, Classification Report)

Visualization: Matplotlib

System Operations: OS, Shutil, Random

# 📦 Dependencies & Requirements

This project requires a Python environment with GPU support (recommended) or CPU. Install the following dependencies:

TensorFlow: pip install tensorflow

Data Handling: pip install numpy scikit-learn

Visualization: pip install matplotlib

# 🧠 Core Logic

The system architecture is engineered for precision and scalability through three distinct phases:

Automated Data Engineering & Augmentation:

Recursive Splitting: A custom Split_Data function performs an 80/20 training-to-testing split, ensuring data integrity by filtering for valid image extensions and utilizing random.shuffle to eliminate selection bias.

Real-time Augmentation: To improve model generalization, the ImageDataGenerator applies rescaling (0-1 normalization), 15-degree rotations, and horizontal flips to simulate various industrial lighting and surface angles.

CNN Architectural Topology:

Feature Extraction: Three hierarchical Conv2D layers (32, 64, and 128 filters) with 3x3 kernels extract spatial features like edges and crack patterns using the ReLU activation function.

Downsampling: MaxPooling2D layers follow each convolution to reduce computational load while retaining critical feature data.

Classification Head: Feature maps are flattened into a 1D vector, passed through a Dense (128 units) layer, and regularized with a Dropout (0.5) layer to prevent overfitting.

Performance Validation:

Optimization: The model is compiled using the Adam optimizer and Binary Cross-Entropy loss.

Metrical Analysis: Success is measured through Confusion Matrices and Classification Reports (Precision, Recall, F1-Score), which are essential for safety-critical inspection systems.

# 🚀 Execution Steps

1. Dataset Initialization

Ensure the raw data is stored in the Concrete Crack Dataset folder with subfolders named Positive and Negative. The script will automatically generate the structured CrackDataset required for the model.

2. Training & Evaluation

Run the main script to start the training process. The model will resize images to 128x128 and process them in batches of 32:

python crack_detection.py


Upon completion, the system generates performance graphs and saves the trained model as Marvellous_Crack_Detection_Model.h5.

3. Single Image Prediction

Use the Predict_Crack function to test unseen images. The function normalizes the input, adds a batch dimension, and outputs the final result:

Predict_Crack("path_to_test_image.jpg")


# 📂 Project Structure

crack_detection.py: Core script for data splitting, training, and evaluation.

# 👤 Author

Sakshi Bhapkar Final-Year Information Technology Engineering Student
