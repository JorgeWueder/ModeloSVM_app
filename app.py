import streamlit as st
import cv2
import pickle
from create_features import FeatureExtractor
from PIL import Image
import numpy as np
import io

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

def mostrar_imagenes(original, resultado, titulo_resultado="Resultado", es_grises=False):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen Original")
        st.image(original, channels="BGR")
    
    with col2:
        st.subheader(titulo_resultado)
        if es_grises:
            # Para imágenes en escala de grises (1 canal)
            st.image(resultado, channels="GRAY")
        else:
            # Para imágenes a color (3 canales)
            st.image(resultado, channels="BGR")

# === Función para Capítulo 1 ===
def ejercicio_capitulo1(imagen):
    """Convierte la imagen a escala de grises"""
    gray_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return gray_img

# === Función para Capítulo 2 ===
def ejercicio_capitulo2(imagen, size=15):
    """Aplica filtro de desenfoque de movimiento"""
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    
    # applying the kernel to the input image
    output = cv2.filter2D(imagen, -1, kernel_motion_blur)
    return output

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
        
        mostrar_imagenes(img, resultado, "Imagen en Escala de Grises", es_grises=True)
        
        # Información adicional
        st.markdown("---")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"**Dimensión original:** {img.shape[1]} x {img.shape[0]} px")
            st.info(f"**Canales original:** {img.shape[2] if len(img.shape) > 2 else 1}")
        with col_info2:
            st.info(f"**Dimensión resultado:** {resultado.shape[1]} x {resultado.shape[0]} px")
            st.info(f"**Canales resultado:** 1 (Escala de grises)")
        
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
    st.header("🌀 Capítulo 2: Filtro de Desenfoque de Movimiento")
    st.write("**Ejercicio:** Aplicación de kernel personalizado para simular desenfoque de movimiento")
    
    # Control deslizante para el tamaño del kernel
    st.subheader("⚙️ Configuración del Filtro")
    kernel_size = st.slider(
        "Tamaño del kernel para el desenfoque:",
        min_value=5,
        max_value=25,
        value=15,
        step=2,
        help="Tamaño impar recomendado para mejor efecto"
    )
    
    img = cargar_imagen()
    if img is not None:
        with st.spinner("Aplicando filtro de desenfoque..."):
            resultado = ejercicio_capitulo2(img, kernel_size)
        
        mostrar_imagenes(img, resultado, f"Desenfoque de Movimiento (Kernel: {kernel_size}x{kernel_size})")
        
        # Información adicional
        st.markdown("---")
        st.subheader("📊 Información del Procesamiento")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.info(f"**Dimensión:** {img.shape[1]} x {img.shape[0]} px")
        with col_info2:
            st.info(f"**Tamaño del kernel:** {kernel_size}x{kernel_size}")
        with col_info3:
            st.info(f"**Tipo de filtro:** Desenfoque horizontal")
        
        # Explicación del kernel
        st.markdown("---")
        st.subheader("🔍 Explicación del Kernel")
        
        # Crear una visualización pequeña del kernel
        kernel_visual = np.zeros((kernel_size, kernel_size))
        kernel_visual[int((kernel_size-1)/2), :] = 1
        
        col_kernel1, col_kernel2 = st.columns([1, 2])
        with col_kernel1:
            st.write("**Kernel utilizado:**")
            st.dataframe(kernel_visual, use_container_width=True)
        
        with col_kernel2:
            st.write("**Descripción:**")
            st.write(f"""
            Este kernel de {kernel_size}x{kernel_size} píxeles crea un efecto de desenfoque de movimiento horizontal:
            - **Fila central:** Todos los valores son 1 (activados)
            - **Otras filas:** Todos los valores son 0 (desactivados)
            - **Normalización:** Todos los valores se dividen por {kernel_size} para mantener el brillo
            """)
        
        # Opción para descargar el resultado
        st.markdown("---")
        st.subheader("💾 Descargar Resultado")
        
        # Convertir para descarga
        result_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        img_bytes = io.BytesIO()
        result_pil.save(img_bytes, format='JPEG')
        
        st.download_button(
            label="Descargar imagen con desenfoque",
            data=img_bytes.getvalue(),
            file_name=f"imagen_desenfoque_movimiento_{kernel_size}.jpg",
            mime="image/jpeg"
        )

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
- ✅ Capítulo 2: Filtro de desenfoque de movimiento
- ✅ Capítulo 9: Clasificación Perros/Gatos
- ⏳ Demás capítulos: Pendientes
""")

# === Pie de página ===
st.markdown("---")
st.markdown("**Desarrollado para el curso de OpenCV** | 📚 11 Capítulos")