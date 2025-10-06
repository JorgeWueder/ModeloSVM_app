import streamlit as st
import cv2
import pickle
from create_features import FeatureExtractor
from PIL import Image
import numpy as np

# === Cargar modelos ===
CODEBOOK_FILE = "models/codebook.pkl"
SVM_FILE = "models/svm_model.pkl"

with open(CODEBOOK_FILE, "rb") as f:
    kmeans, centroids = pickle.load(f)

with open(SVM_FILE, "rb") as f:
    svm = pickle.load(f)

# === Definir el clasificador ===
class ImageClassifier:
    def __init__(self, svm, kmeans, centroids):
        self.svm = svm
        self.kmeans = kmeans
        self.centroids = centroids
        self.extractor = FeatureExtractor()

    def predict(self, img_array):
        fv = self.extractor.get_feature_vector(img_array, self.kmeans, self.centroids)
        return self.svm.classify(fv)[0]  # Devuelve la etiqueta como 'perro' o 'gato'

classifier = ImageClassifier(svm, kmeans, centroids)

# === Streamlit UI ===
st.title("Clasificador de Perros y Gatos 🐶🐱")

uploaded_file = st.file_uploader("Sube una imagen de prueba", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abrir imagen
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Imagen subida", use_column_width=True)

    # Predicción
    label = classifier.predict(image_np)
    st.success(f"Predicción: {label}")
