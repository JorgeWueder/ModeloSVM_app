import streamlit as st
import cv2
import pickle
from create_features import FeatureExtractor
from PIL import Image
import numpy as np

# === Configuración de la página ===
st.set_page_config(
    page_title="Ejercicios OpenCV - 11 Capítulos",
    page_icon="🎨",
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

# === Función para Capítulo 3 - CORREGIDA ===
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
 
    # CORRECCIÓN: Asegurar que mask tenga las mismas dimensiones que img_output
    mask_resized = cv2.resize(mask, (img_output.shape[1], img_output.shape[0]))
    
    # Convertir mask a 3 canales para poder hacer bitwise_and con img_output
    mask_3d = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    
    # Add the thick boundary lines to the image using 'AND' operator 
    dst = cv2.bitwise_and(img_output, mask_3d) 
    return dst

# === Función para Capítulo 4 ===
def ejercicio_capitulo4(imagen):
    # Cargar el clasificador de rostros desde tu archivo específico
    face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
    
    # Verificar si se cargó correctamente
    if face_cascade.empty():
        st.error("❌ No se pudo cargar el clasificador de rostros. Verifica que el archivo exista en 'cascade_files/'")
        return imagen, 0
    
    # Convertir a escala de grises para la detección
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    # Dibujar rectángulos alrededor de los rostros detectados
    resultado = imagen.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(resultado, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    return resultado, len(faces)

# === Función para Capítulo 5 ===
def ejercicio_capitulo5(imagen, max_corners=7, quality_level=0.05, min_distance=25):
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detectar esquinas
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, 
                                    qualityLevel=quality_level, 
                                    minDistance=min_distance)
    
    # Dibujar círculos en las esquinas detectadas
    resultado = imagen.copy()
    if corners is not None:
        corners = np.float32(corners)
        for item in corners:
            x, y = item[0]
            cv2.circle(resultado, (int(x), int(y)), 5, (0, 0, 255), -1)  # Círculos rojos
    
    return resultado, len(corners) if corners is not None else 0

# === Sidebar para navegación ===
st.sidebar.title("🎯 Navegación")
capitulo = st.sidebar.selectbox(
    "Selecciona un capítulo:",
    [
        "🏠 Introducción", 
        "📷 Capítulo 1", "🌀 Capítulo 2", "🎨 Capítulo 3", "👤 Capítulo 4", "🔺 Capítulo 5",
        "⚡ Capítulo 6", "🎯 Capítulo 7", "🌟 Capítulo 8", "🐱 Capítulo 9", "🚀 Capítulo 10", "💫 Capítulo 11"
    ]
)

# === Contenido principal ===
st.title("🎨 Ejercicios de OpenCV - 11 Capítulos")

if capitulo == "🏠 Introducción":
    st.header("🎉 Bienvenido")
    st.write("Selecciona un capítulo en el sidebar")
    
elif capitulo == "📷 Capítulo 1":
    st.header("📷 Capítulo 1: Conversión a Escala de Grises")
    st.write("**Qué hace:** Convierte una imagen a color a escala de grises")
    
    img = cargar_imagen()
    if img is not None:
        resultado = ejercicio_capitulo1(img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="Original")
        with col2:
            st.image(resultado, caption="Escala de Grises", use_column_width=True)

elif capitulo == "🌀 Capítulo 2":
    st.header("🌀 Capítulo 2: Filtro de Desenfoque de Movimiento")
    st.write("**Qué hace:** Aplica un filtro que simula desenfoque por movimiento horizontal")
    
    kernel_size = st.slider("Tamaño del kernel:", 5, 25, 15, 2)
    img = cargar_imagen()
    if img is not None:
        resultado = ejercicio_capitulo2(img, kernel_size)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="Original")
        with col2:
            st.image(resultado, channels="BGR", caption=f"Desenfoque (Kernel: {kernel_size})")

elif capitulo == "🎨 Capítulo 3":
    st.header("🎨 Capítulo 3: Efecto Cartoon")
    st.write("**Qué hace:** Transforma imágenes en estilo cartoon o sketch")
    
    # Selector de modo
    modo = st.radio(
        "🎭 Selecciona el modo:",
        ["🖼️ Original", "🌈 Cartoon con Color", "✏️ Sketch (Sin Color)"]
    )
    
    img = cargar_imagen()
    if img is not None:
        if modo == "🖼️ Original":
            resultado = img
        elif modo == "🌈 Cartoon con Color":
            resultado = cartoonize_image(img, ksize=5, sketch_mode=False)
        else:  # Sketch (Sin Color)
            resultado = cartoonize_image(img, ksize=5, sketch_mode=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="🖼️ Imagen Original")
        with col2:
            st.image(resultado, channels="BGR", caption=modo)

elif capitulo == "👤 Capítulo 4":
    st.header("👤 Capítulo 4: Detección de Rostros")
    st.write("**Qué hace:** Detecta rostros humanos en imágenes usando el clasificador Haar Cascade")
    
    img = cargar_imagen()
    if img is not None:
        with st.spinner("🔍 Detectando rostros..."):
            resultado, num_faces = ejercicio_capitulo4(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="🖼️ Imagen Original")
        with col2:
            st.image(resultado, channels="BGR", caption=f"👤 Rostros detectados: {num_faces}")
        
        # Mostrar información adicional
        if num_faces > 0:
            st.success(f"✅ Se detectaron {num_faces} rostro(s) en la imagen")
        else:
            st.warning("⚠️ No se detectaron rostros en la imagen")

elif capitulo == "🔺 Capítulo 5":
    st.header("🔺 Capítulo 5: Detección de Esquinas")
    st.write("**Qué hace:** Detecta esquinas en imágenes usando el algoritmo Good Features to Track")
    
    # Controles para los parámetros
    col_params1, col_params2, col_params3 = st.columns(3)
    with col_params1:
        max_corners = st.slider("Máximo de esquinas:", 1, 50, 7, 1)
    with col_params2:
        quality_level = st.slider("Nivel de calidad:", 0.01, 0.2, 0.05, 0.01)
    with col_params3:
        min_distance = st.slider("Distancia mínima:", 5, 50, 25, 5)
    
    img = cargar_imagen()
    if img is not None:
        with st.spinner("🔍 Detectando esquinas..."):
            resultado, num_corners = ejercicio_capitulo5(img, max_corners, quality_level, min_distance)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="🖼️ Imagen Original")
        with col2:
            st.image(resultado, channels="BGR", caption=f"🔺 Esquinas detectadas: {num_corners}")
        
        # Mostrar información adicional
        if num_corners > 0:
            st.success(f"✅ Se detectaron {num_corners} esquina(s) en la imagen")
        else:
            st.warning("⚠️ No se detectaron esquinas en la imagen")

elif capitulo == "⚡ Capítulo 6":
    st.header("⚡ Capítulo 6")
    st.write("**Qué hace:** [Descripción pendiente]")
    img = cargar_imagen()
    if img is not None:
        st.info("⏳ Pendiente: Integrar código del Capítulo 6")

elif capitulo == "🎯 Capítulo 7":
    st.header("🎯 Capítulo 7")
    st.write("**Qué hace:** [Descripción pendiente]")
    img = cargar_imagen()
    if img is not None:
        st.info("⏳ Pendiente: Integrar código del Capítulo 7")

elif capitulo == "🌟 Capítulo 8":
    st.header("🌟 Capítulo 8")
    st.write("**Qué hace:** [Descripción pendiente]")
    img = cargar_imagen()
    if img is not None:
        st.info("⏳ Pendiente: Integrar código del Capítulo 8")

elif capitulo == "🐱 Capítulo 9":
    st.header("🐱 Capítulo 9: Clasificación Perros vs Gatos")
    st.write("**Qué hace:** Clasifica imágenes entre perros y gatos usando Machine Learning")
    
    if classifier is None:
        st.error("❌ No se pudieron cargar los modelos")
    else:
        uploaded_file = st.file_uploader("Sube una imagen de perro o gato", type=["jpg", "jpeg", "png"], key="cap9")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="🖼️ Imagen Original")
            
            with col2:
                with st.spinner("🔍 Clasificando..."):
                    label = classifier.predict(image_np)
                if "perro" in label.lower():
                    st.success(f"🐶 **Predicción: {label}**")
                else:
                    st.success(f"🐱 **Predicción: {label}**")

elif capitulo == "🚀 Capítulo 10":
    st.header("🚀 Capítulo 10")
    st.write("**Qué hace:** [Descripción pendiente]")
    img = cargar_imagen()
    if img is not None:
        st.info("⏳ Pendiente: Integrar código del Capítulo 10")

elif capitulo == "💫 Capítulo 11":
    st.header("💫 Capítulo 11")
    st.write("**Qué hace:** [Descripción pendiente]")
    img = cargar_imagen()
    if img is not None:
        st.info("⏳ Pendiente: Integrar código del Capítulo 11")

# === Información en el sidebar ===
st.sidebar.markdown("---")
st.sidebar.info("""
**📊 Estado:**
- ✅ Capítulo 1: Escala de grises
- ✅ Capítulo 2: Desenfoque movimiento  
- ✅ Capítulo 3: Efecto cartoon
- ✅ Capítulo 4: Detección de rostros
- ✅ Capítulo 5: Detección de esquinas
- ✅ Capítulo 9: Clasificación Perros/Gatos
- ⏳ Demás capítulos: Pendientes
""")