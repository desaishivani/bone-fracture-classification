# 🦴 Bone Fracture Classification

A deep learning model that classifies bone fracture types from X-ray images.

## 📁 Dataset
- Source: [Kaggle - Bone Break Classification](https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset)
- 10 Fracture types:
  - Avulsion, Comminuted, Fracture Dislocation, Greenstick, Hairline, Impacted, Longitudinal, Oblique, Pathological, Spiral

## ✅ Tasks Completed

### 1. Data Preprocessing
- Resized all images to 224×224
- Normalized pixel values
- Applied data augmentation

### 2. Model Training
- Used MobileNetV2 with transfer learning
- Accuracy improved after augmentation
- Final test accuracy: **~46%**

### 3. Streamlit App
- User uploads an X-ray image
- App predicts fracture type and confidence score
- Uses saved `.keras` model for inference

### 4. Project Files
- `Bone_Classification.ipynb`: All training steps
- `mobilenet_fracture_model_da.keras`: Final model
- `app.py`: Streamlit app
- `README.md`: Project overview

## 🧪 To Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
