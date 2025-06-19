import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model = tf.keras.models.load_model("mobilenet_fracture_model_da.keras")

# Class labels
class_labels = ['Avulsion fracture', 'Comminuted fracture', 'Fracture Dislocation', 'Greenstick fracture',
                'Hairline Fracture', 'Impacted fracture', 'Longitudinal fracture', 'Oblique fracture',
                'Pathological fracture', 'Spiral Fracture']

# UI
st.title(" Bone Fracture Classifier")
st.write("Upload an X-ray image to classify the fracture type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))

    # Predict
    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions)
    pred_class = class_labels[pred_idx]
    confidence = predictions[0][pred_idx]

    # Display
    st.markdown(f"### Prediction: `{pred_class}`")
    st.markdown(f"Confidence: `{confidence:.2f}`")
