# 🌿 Plant Disease Detection

A deep learning–powered web application that detects diseases in plant leaves from uploaded images. Built as a **Capstone 2 group project**, this system helps farmers, gardeners, and agricultural researchers quickly identify plant health issues using a trained convolutional neural network.

---

## How It Works

1. The user uploads a photo of a plant leaf through the web interface.
2. The image is preprocessed and fed into a TensorFlow/Keras CNN model trained on the [PlantVillage dataset](https://drive.google.com/drive/folders/1d3kmQlD2J1hGHqd09-rVo_UpvTZ8nPQF?usp=sharing).
3. The model predicts the plant species and disease condition with a confidence score.
4. The app returns the top predictions along with a human-readable diagnosis and recommended next steps.

---

## Features

- **Image Upload & Analysis** — Upload any leaf photo and get an instant prediction
- **Multi-class Classification** — Identifies diseases across multiple plant species (apple, tomato, corn, and more)
- **Top-K Predictions** — Returns the top 8 most likely diagnoses with confidence percentages
- **AI-Generated Diagnosis** — Provides a plain-language summary of the condition and actionable advice
- **Healthy vs. Diseased Detection** — Clearly flags whether the leaf is healthy or shows disease symptoms
- **REST API Backend** — Flask-based API with a `/analyze` endpoint for easy integration

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python, Flask, Flask-CORS |
| **ML Framework** | TensorFlow / Keras |
| **Image Processing** | Pillow (PIL), NumPy |
| **Model Format** | TensorFlow SavedModel (`.keras`) |
| **Dataset** | PlantVillage (via Google Drive) |

---

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Dev2316/Plant-Disease-Detection.git
cd Plant-Disease-Detection

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`.

### Training the Model (Optional)

```bash
# Train from scratch
python train_plantvillage_model.py

# Or train using transfer learning
python train_plantvillage_transfer.py
```

---

## Project Structure

```
Plant-Disease-Detection/
├── app.py                        # Flask backend & REST API
├── index.html                    # Frontend UI
├── train_plantvillage_model.py   # Model training script
├── train_plantvillage_transfer.py # Transfer learning training script
├── plant_disease_model.keras     # Pre-trained model file
├── FINAL_CODE.ipynb              # Jupyter notebook (experiments & analysis)
├── requirements.txt              # Python dependencies
└── link to dataset.txt           # Link to the PlantVillage dataset
```

---

## Dataset

The model is trained on the **PlantVillage dataset**, which contains labeled images of healthy and diseased leaves across 14+ plant species and 38 disease categories.

📂 [Download the dataset here](https://drive.google.com/drive/folders/1d3kmQlD2J1hGHqd09-rVo_UpvTZ8nPQF?usp=sharing)

---

## 👥 Team Members

| Name |
|---|
| Dev R Patel |
| Dev S Patel |
| Maharshi Vyas |
| Parth Modi |
| Harsh Patel |

---

## Project Status

> This project is currently a work in progress. Additional features and improvements are planned.

---

## 📄 License

This project was developed as an academic capstone project. Feel free to fork and build upon it for educational purposes.
