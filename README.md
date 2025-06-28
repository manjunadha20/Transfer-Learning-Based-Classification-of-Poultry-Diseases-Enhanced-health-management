# 🐔 Poultry Disease Classification Web App

This is a deep learning-based web application for **detecting poultry diseases** from images using a Convolutional Neural Network (MobileNetV2). It classifies images into four categories: **Coccidiosis, Healthy, Newcastle, and Salmonella**, and provides treatment suggestions.

---

## 📁 Folder Structure

poultry_web_app/
├── app.py # Flask application
├── train_model.py # Model training script
├── poultry_disease_model.h5 # Trained model (after training)
├── templates/
│ └── index.html # Frontend HTML
├── static/
│ └── style.css # Custom CSS
├── data/
│ └── data/
│ ├── train/
│ │ ├── Coccidiosis/
│ │ ├── Healthy/
│ │ ├── Newcastle/
│ │ └── Salmonella/
│ ├── val/
│ │ ├── Coccidiosis/
│ │ ├── Healthy/
│ │ ├── Newcastle/
│ │ └── Salmonella/
│ └── test/
│ ├── Coccidiosis/
│ ├── Healthy/
│ ├── Newcastle/
│ └── Salmonella/
├── trim_dataset.py # Script to reduce dataset size
├── README.md # This file


## ⚙️ Features

- 🧠 Trained on **MobileNetV2** (transfer learning)
- 📷 Image upload and classification
- 💊 Treatment suggestions for each disease
- 💻 Responsive web app UI with custom CSS and background
- ✅ Trained locally with GPU (NVIDIA GeForce GTX)
- ✂️ Optionally trimmed dataset to 300 images per class

Dataset Source:https://www.kaggle.com/datasets/chandrashekarnatesh/poultry-diseases

## 🚀 How to Run

### 1. 🛠️ Install Required Packages

Make sure you have Python 3.x installed. Then:

```bash
pip install tensorflow flask pillow numpy
2. 🧪 Train the Model
Run this to train on your dataset:

python train_model.py
This will create poultry_disease_model.h5.

⚠️ Already trained? You can skip this step if the file exists.

3. 🖼️ (Optional) Trim Dataset
If your dataset is large, reduce it to 300 images per class for faster training:


python trim_dataset.py
4. 🌐 Run the Web App

python app.py
Visit: http://127.0.0.1:5000

🖼️ Image Classification & Treatment
The app allows users to upload a poultry image and get a prediction. Each prediction is accompanied by treatment suggestions:

🐓 Disease	💊 Treatment Suggestion
Coccidiosis	Use anticoccidial drugs (e.g., amprolium). Keep litter dry and clean.
Newcastle	Isolate infected birds. Vaccinate healthy ones. Disinfect surroundings.
Salmonella	Administer antibiotics (e.g., enrofloxacin) under vet supervision.
Healthy	No treatment needed. Keep monitoring. ✅

## 🔍 App Screenshots

### 📸 Screenshot 1:
![App Screenshot 1](static/app_screenshot1.PNG)

### 📸 Screenshot 2:
![App Screenshot 2](static/app_screenshot2.png)


🧪 Model Used
MobileNetV2 pretrained on ImageNet

Input image size: 224x224

Output: 4 classes with softmax activation

🩺 Disease Treatment Suggestions
Disease	Treatment
Coccidiosis	Use anticoccidial drugs like amprolium. Keep litter dry and sanitized.
Newcastle	Isolate infected birds, disinfect area, and vaccinate healthy birds.
Salmonella	Use antibiotics like enrofloxacin (under vet supervision). Ensure clean feed and water.
Healthy	No treatment needed. Keep monitoring. ✅

🎨 UI Highlights
Centered upload and prediction area

Blue buttons with hover effects

Background image set via static/style.css

Clean minimalistic HTML structure

👩‍💻 Author
Ujwala Perugu

📌 Note
This is a development version. For production, consider deploying with gunicorn or Docker.

The model was trained on a reduced dataset. For higher accuracy, train on the full dataset.
