## Face Mask Detection System
This project is a deep learning-based computer vision application designed to detect face masks in real-time. It features a robust PyTorch backend, a Flask API for deployment, and a responsive frontend for user interaction.

## 🚀 Features
  Custom CNN Architectures: Includes Basic, Optimized, and Enhanced CNN models.
  Advanced Training: Implements Label Smoothing, Cosine Annealing, and Early Stopping to ensure high accuracy.
  Real-time Detection: Flask-based API supporting both static image uploads and real-time webcam streaming.
  Performance Analysis: Automated generation of Confusion Matrices and Loss/Accuracy curves.

## 📁 Project Structure
  CV_Project/
      ├── backend/                # Model logic and API services
      │   ├── models/             # Saved model weights (.pth)
      │   ├── api.py              # Flask REST API
      │   ├── train.py            # Training and evaluation script
      │   └── model_architecture.py # CNN class definitions
      ├── frontend/               # Web-based user interface          
      │     ├── index             # Main web page      
      │     ├── script.js         # Frontend Logic
      │     ├── style.css         # Styling 
      ├── .gitignore              # Files excluded from GitHub (data)
      ├── requirements.txt        # List of python libraries
      └── README.md               # Project documentation

## 🛠️ Tech Stack
  Deep Learning: PyTorch, Torchvision.
  Backend: Flask (Python).
  Image Processing: OpenCV, PIL.
  Analysis: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn.

## ⚙️ Installation & Setup
  1.Clone the Repository:
    git clone https://github.com/LeqaaBedair0/Face-Mask-Detection.git
    cd CV_Project
  2.Install Dependencies:
    pip install -r requirements.txt
  3.Start the Backend API:
    python backend/api.py
  4.Launch the Frontend:
    Simply open frontend/index.html in any modern web browser to interact with the model.

## 📊 Model Performance
  The EnhancedCNN model utilizes Batch Normalization and Dropout (0.4) to stabilize training and minimize overfitting. The training process uses Data Augmentation (rotation, flipping, and random erasing) to ensure the system remains accurate across various environments.
    Dataset Note: The image dataset is excluded from this repository via .gitignore to keep the repo lightweight. You can use the Face Mask Detection Dataset from Kaggle for retraining.





      
