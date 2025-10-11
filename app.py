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

def mostrar_imagenes(original, resultado, titulo_resultado="Resultado"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen Original")
        st.image(original, channels="BGR")
    with col2:
        st.subheader(titulo_resultado)
        st.image(resultado, channels="BGR")

# === Función para Capítulo 1 ===
def ejercicio_capitulo1(imagen):
    """Convierte la imagen a escala de grises"""
    gray_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return gray_img

# === Sidebar para navegación ===
st.sidebar.title("📚 Navegación de Capítulos")
capitulo = st.sidebar.selectbox(
    "Selecciona un capítulo:",
    [
        "Introducción", 
        "Capítulo 1", "Capítulo 2", "Capítulo 3", "Capítulo 4", "Capítulo 5",
        "Capítulo 6", "Capítulo 7", "Capítulo 8", "Capítulo 9", "Capítulo 10", "Capítulo 11"
    ]
)

# === Contenido principal ===
st.title("📚 Ejercicios de OpenCV - 11 Capítulos")
st.markdown("---")

if capitulo == "Introducción":
    st.header("Bienvenido a los Ejercicios de OpenCV")
    st.write("""
    Esta aplicación contiene los ejercicios de los 11 capítulos de OpenCV.
    
    **Instrucciones:**
    1. Selecciona un capítulo en el sidebar
    2. Sube una imagen para probar los algoritmos
    3. Explora los diferentes resultados
    """)
    
elif capitulo == "Capítulo 1":
    st.header("🎯 Capítulo 1: Introducción a OpenCV")
    st.write("**Ejercicio:** Conversión de imagen a escala de grises")
    
    img = cargar_imagen()
    if img is not None:
        with st.spinner("Procesando imagen..."):
            resultado = ejercicio_capitulo1(img)
        
        mostrar_imagenes(img, resultado, "Imagen en Escala de Grises")
        
        # Información adicional
        st.markdown("---")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"**Dimensión original:** {img.shape[1]} x {img.shape[0]} px")
        with col_info2:
            st.info(f"**Dimensión resultado:** {resultado.shape[1]} x {resultado.shape[0]} px")
        
        # Opción para descargar el resultado
        st.markdown("---")
        st.subheader("💾 Descargar Resultado")
        # Convertir para descarga
        result_pil = Image.fromarray(resultado)
        img_bytes = io.BytesIO()
        result_pil.save(img_bytes, format='JPEG')
        
        st.download_button(
            label="Descargar imagen en escala de grises",
            data=img_bytes.getvalue(),
            file_name="imagen_escala_grises.jpg",
            mime="image/jpeg"
        )

elif capitulo == "Capítulo 2":
    st.header("Capítulo 2: Operaciones Básicas")
    st.write("Aquí va tu ejercicio del capítulo 2")
    img = cargar_imagen()
    if img is not None:
        # ESPACIO PARA TU CÓDIGO DEL CAPÍTULO 2
        st.info("Pendiente: Integrar código del Capítulo 2")

elif capitulo == "Capítulo 3":
    st.header("Capítulo 3: Transformaciones de Imagen")
    st.write("Aquí va tu ejercicio del capítulo 3")
    img = cargar_imagen()
    if img is not None:
        # ESPACIO PARA TU CÓDIGO DEL CAPÍTULO 3
        st.info("Pendiente: Integrar código del Capítulo 3")

elif capitulo == "Capítulo 4":
    st.header("Capítulo 4: Filtros y Convolución")
    st.write("Aquí va tu ejercicio del capítulo 4")
    img = cargar_imagen()
    if img is not None:
        # ESPACIO PARA TU CÓDIGO DEL CAPÍTULO 4
        st.info("Pendiente: Integrar código del Capítulo 4")

elif capitulo == "Capítulo 5":
    st.header("Capítulo 5: Detección de Bordes")
    st.write("Aquí va tu ejercicio del capítulo 5")
    img = cargar_imagen()
    if img is not None:
        # ESPACIO PARA TU CÓDIGO DEL CAPÍTULO 5
        st.info("Pendiente: Integrar código del Capítulo 5")

elif capitulo == "Capítulo 6":
    st.header("Capítulo 6: Transformaciones Morfológicas")
    st.write("Aquí va tu ejercicio del capítulo 6")
    img = cargar_imagen()
    if img is not None:
        # ESPACIO PARA TU CÓDIGO DEL CAPÍTULO 6
        st.info("Pendiente: Integrar código del Capítulo 6")

elif capitulo == "Capítulo 7":
    st.header("Capítulo 7: Segmentación de Imágenes")
    st.write("Aquí va tu ejercicio del capítulo 7")
    img = cargar_imagen()
    if img is not None:
        # ESPACIO PARA TU CÓDIGO DEL CAPÍTULO 7
        st.info("Pendiente: Integrar código del Capítulo 7")

elif capitulo == "Capítulo 8":
    st.header("Capítulo 8: Detección de Características")
    st.write("Aquí va tu ejercicio del capítulo 8")
    img = cargar_imagen()
    if img is not None:
        # ESPACIO PARA TU CÓDIGO DEL CAPÍTULO 8
        st.info("Pendiente: Integrar código del Capítulo 8")

elif capitulo == "Capítulo 9":
    st.header("🐶🐱 Capítulo 9: Clasificación de Imágenes (Perros vs Gatos)")
    
    if classifier is None:
        st.error("❌ No se pudieron cargar los modelos. Verifica que los archivos existan en la carpeta 'models/'")
    else:
        uploaded_file = st.file_uploader("Sube una imagen de perro o gato", type=["jpg", "jpeg", "png"], key="cap9")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagen Original")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Predicción")
                with st.spinner("Clasificando..."):
                    label = classifier.predict(image_np)
                
                if "perro" in label.lower():
                    st.success(f"🐶 **Predicción: {label}**")
                else:
                    st.success(f"🐱 **Predicción: {label}**")

elif capitulo == "Capítulo 10":
    st.header("Capítulo 10: Procesamiento Avanzado")
    st.write("Aquí va tu ejercicio del capítulo 10")
    img = cargar_imagen()
    if img is not None:
        # ESPACIO PARA TU CÓDIGO DEL CAPÍTULO 10
        st.info("Pendiente: Integrar código del Capítulo 10")

elif capitulo == "Capítulo 11":
    st.header("Capítulo 11: Aplicaciones Finales")
    st.write("Aquí va tu ejercicio del capítulo 11")
    img = cargar_imagen()
    if img is not None:
        # ESPACIO PARA TU CÓDIGO DEL CAPÍTULO 11
        st.info("Pendiente: Integrar código del Capítulo 11")

# === Información en el sidebar ===
st.sidebar.markdown("---")
st.sidebar.info("""
**Estado:**
- ✅ Capítulo 1: Conversión a escala de grises
- ✅ Capítulo 9: Clasificación Perros/Gatos
- ⏳ Demás capítulos: Pendientes
""")

# === Pie de página ===
st.markdown("---")
st.markdown("**Desarrollado para el curso de OpenCV** | 📚 11 Capítulos")