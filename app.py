import streamlit as st
import cv2
import pickle
from create_features import FeatureExtractor
from PIL import Image
import numpy as np

# === Configuración de la página ===
st.set_page_config(
    page_title="Ejercicios OpenCV - 11 Capítulos",
    page_icon="📚",
    layout="wide"
)

# === Cargar modelos (para Capítulo 9) ===
@st.cache_resource
def cargar_modelos():
    try:
        CODEBOOK_FILE = "models/codebook.pkl"
        SVM_FILE = "models/svm_model.pkl"
        
        with open(CODEBOOK_FILE, "rb") as f:
            kmeans, centroids = pickle.load(f)

        with open(SVM_FILE, "rb") as f:
            svm = pickle.load(f)
            
        return kmeans, centroids, svm
    except:
        return None, None, None

kmeans, centroids, svm = cargar_modelos()

# === Definir el clasificador (Capítulo 9) ===
class ImageClassifier:
    def __init__(self, svm, kmeans, centroids):
        self.svm = svm
        self.kmeans = kmeans
        self.centroids = centroids
        self.extractor = FeatureExtractor()

    def predict(self, img_array):
        fv = self.extractor.get_feature_vector(img_array, self.kmeans, self.centroids)
        return self.svm.classify(fv)[0]

classifier = ImageClassifier(svm, kmeans, centroids) if svm else None

# === Funciones auxiliares ===
def cargar_imagen():
    uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return None

# === Función para Capítulo 1 ===
def ejercicio_capitulo1(imagen):
    gray_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return gray_img

# === Función para Capítulo 2 ===
def ejercicio_capitulo2(imagen, size=15):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    output = cv2.filter2D(imagen, -1, kernel_motion_blur)
    return output

# === Función para Capítulo 3 ===
def cartoonize_image(img, ksize=5, sketch_mode=False):
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4 
    # Convert image to grayscale 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
    # Apply median filter to the grayscale image 
    img_gray = cv2.medianBlur(img_gray, 7) 
 
    # Detect edges in the image and threshold it 
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize) 
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV) 
 
    # 'mask' is the sketch of the image 
    if sketch_mode: 
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
 
    # Resize the image to a smaller size for faster computation 
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
 
    # Apply bilateral filter the image multiple times 
    for i in range(num_repetitions): 
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space) 
 
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR) 
 
    dst = np.zeros(img_gray.shape) 
 
    # Add the thick boundary lines to the image using 'AND' operator 
    dst = cv2.bitwise_and(img_output, img_output, mask=mask) 
    return dst

# === Sidebar para navegación ===
st.sidebar.title("Navegación")
capitulo = st.sidebar.selectbox(
    "Selecciona un capítulo:",
    [
        "Introducción", 
        "Capítulo 1", "Capítulo 2", "Capítulo 3", "Capítulo 4", "Capítulo 5",
        "Capítulo 6", "Capítulo 7", "Capítulo 8", "Capítulo 9", "Capítulo 10", "Capítulo 11"
    ]
)

# === Contenido principal ===
st.title("Ejercicios de OpenCV - 11 Capítulos")

if capitulo == "Introducción":
    st.header("Bienvenido")
    st.write("Selecciona un capítulo en el sidebar")
    
elif capitulo == "Capítulo 1":
    st.header("Capítulo 1: Escala de Grises")
    img = cargar_imagen()
    if img is not None:
        resultado = ejercicio_capitulo1(img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="Original")
        with col2:
            st.image(resultado, caption="Escala de Grises", use_column_width=True)

elif capitulo == "Capítulo 2":
    st.header("Capítulo 2: Desenfoque de Movimiento")
    kernel_size = st.slider("Tamaño del kernel:", 5, 25, 15, 2)
    img = cargar_imagen()
    if img is not None:
        resultado = ejercicio_capitulo2(img, kernel_size)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="Original")
        with col2:
            st.image(resultado, channels="BGR", caption=f"Desenfoque (Kernel: {kernel_size})")

elif capitulo == "Capítulo 3":
    st.header("Capítulo 3: Cartoonizado de Imágenes")
    
    # Selector de modo
    modo = st.radio(
        "Selecciona el modo de cartoonizado:",
        ["Original", "Cartoon con Color", "Sketch (Sin Color)"]
    )
    
    img = cargar_imagen()
    if img is not None:
        if modo == "Original":
            resultado = img
        elif modo == "Cartoon con Color":
            resultado = cartoonize_image(img, ksize=5, sketch_mode=False)
        else:  # Sketch (Sin Color)
            resultado = cartoonize_image(img, ksize=5, sketch_mode=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="Imagen Original")
        with col2:
            st.image(resultado, channels="BGR", caption=f"Modo: {modo}")

elif capitulo == "Capítulo 4":
    st.header("Capítulo 4")
    img = cargar_imagen()
    if img is not None:
        st.info("Pendiente: Integrar código del Capítulo 4")

elif capitulo == "Capítulo 5":
    st.header("Capítulo 5")
    img = cargar_imagen()
    if img is not None:
        st.info("Pendiente: Integrar código del Capítulo 5")

elif capitulo == "Capítulo 6":
    st.header("Capítulo 6")
    img = cargar_imagen()
    if img is not None:
        st.info("Pendiente: Integrar código del Capítulo 6")

elif capitulo == "Capítulo 7":
    st.header("Capítulo 7")
    img = cargar_imagen()
    if img is not None:
        st.info("Pendiente: Integrar código del Capítulo 7")

elif capitulo == "Capítulo 8":
    st.header("Capítulo 8")
    img = cargar_imagen()
    if img is not None:
        st.info("Pendiente: Integrar código del Capítulo 8")

elif capitulo == "Capítulo 9":
    st.header("Capítulo 9: Clasificación Perros vs Gatos")
    
    if classifier is None:
        st.error("No se pudieron cargar los modelos")
    else:
        uploaded_file = st.file_uploader("Sube una imagen de perro o gato", type=["jpg", "jpeg", "png"], key="cap9")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Imagen Original")
            
            with col2:
                label = classifier.predict(image_np)
                st.success(f"Predicción: {label}")

elif capitulo == "Capítulo 10":
    st.header("Capítulo 10")
    img = cargar_imagen()
    if img is not None:
        st.info("Pendiente: Integrar código del Capítulo 10")

elif capitulo == "Capítulo 11":
    st.header("Capítulo 11")
    img = cargar_imagen()
    if img is not None:
        st.info("Pendiente: Integrar código del Capítulo 11")