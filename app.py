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

# === FUNCIONES PARA CAPÍTULO 6 - Seam Carving CORREGIDO ===
def overlay_vertical_seam(img, seam): 
    img_seam_overlay = np.copy(img)
    x_coords, y_coords = np.transpose([(i,int(j)) for i,j in enumerate(seam)]) 
    img_seam_overlay[x_coords, y_coords] = (0,255,0) 
    return img_seam_overlay

def compute_energy_matrix(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) 
    abs_sobel_x = cv2.convertScaleAbs(sobel_x) 
    abs_sobel_y = cv2.convertScaleAbs(sobel_y) 
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0) 

def find_vertical_seam(img, energy): 
    rows, cols = img.shape[:2] 
    seam = np.zeros(img.shape[0]) 
    dist_to = np.zeros(img.shape[:2]) + float('inf') 
    dist_to[0,:] = np.zeros(img.shape[1]) 
    edge_to = np.zeros(img.shape[:2]) 

    for row in range(rows-1): 
        for col in range(cols): 
            if col != 0 and \
            dist_to[row+1, col-1] > dist_to[row, col] + energy[row+1, col-1]: 
                dist_to[row+1, col-1] = dist_to[row, col] + energy[row+1, col-1] 
                edge_to[row+1, col-1] = 1 

            if dist_to[row+1, col] > dist_to[row, col] + energy[row+1, col]: 
                dist_to[row+1, col] = dist_to[row, col] + energy[row+1, col] 
                edge_to[row+1, col] = 0 

            if col != cols-1: 
                if dist_to[row+1, col+1] > dist_to[row, col] + energy[row+1, col+1]: 
                    dist_to[row+1, col+1] = dist_to[row, col] + energy[row+1, col+1] 
                    edge_to[row+1, col+1] = -1

    seam[rows-1] = np.argmin(dist_to[rows-1, :]) 
    for i in (x for x in reversed(range(rows)) if x > 0): 
        seam[i-1] = seam[i] + edge_to[i, int(seam[i])] 

    return seam 

def add_vertical_seam(img, seam, num_iter): 
    seam = seam + num_iter 
    rows, cols = img.shape[:2] 
    zero_col_mat = np.zeros((rows,1,3), dtype=np.uint8) 
    img_extended = np.hstack((img, zero_col_mat)) 

    for row in range(rows): 
        for col in range(cols, int(seam[row]), -1): 
            img_extended[row, col] = img[row, col-1] 

        for i in range(3): 
            v1 = img_extended[row, int(seam[row])-1, i] 
            v2 = img_extended[row, int(seam[row])+1, i] 
            img_extended[row, int(seam[row]), i] = (int(v1)+int(v2))/2 

    return img_extended 

def remove_vertical_seam(img, seam): 
    rows, cols = img.shape[:2] 
    for row in range(rows): 
        for col in range(int(seam[row]), cols-1): 
            img[row, col] = img[row, col+1] 

    img = img[:, 0:cols-1] 
    return img 

def ejercicio_capitulo6(imagen, num_seams, modo):
    img = np.copy(imagen)
    img_output = np.copy(imagen)
    img_overlay_seam = np.copy(imagen)
    
    energy = compute_energy_matrix(img)
    
    for i in range(num_seams):
        seam = find_vertical_seam(img, energy)
        img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
        
        if modo == "eliminar":
            img = remove_vertical_seam(img, seam)
            energy = compute_energy_matrix(img)
        else:  # agregar
            img_output = add_vertical_seam(img_output, seam, i)
    
    if modo == "eliminar":
        return img, img_overlay_seam
    else:
        return img_output, img_overlay_seam

# === FUNCIONES PARA CAPÍTULO 7 - Defectos de Convexidad CORREGIDO ===
def get_all_contours(img):
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def ejercicio_capitulo7(imagen, factor_epsilon=0.01):
    img_resultado = np.copy(imagen)
    total_defectos = 0
    
    # Iterar sobre los contornos extraídos
    for contour in get_all_contours(imagen):
        if len(contour) < 3:  # Necesitamos al menos 3 puntos para un contorno válido
            continue
            
        orig_contour = contour
        epsilon = factor_epsilon * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Verificar que el contorno aproximado tenga suficientes puntos
        if len(contour) < 3:
            continue
            
        # Extraer casco convexo y defectos de convexidad
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            
            # Para convexityDefects necesitamos al menos 3 puntos en el hull
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                
                if defects is not None:
                    for i in range(defects.shape[0]):
                        start_defect, end_defect, far_defect, distance = defects[i, 0]
                        
                        # Filtrar defectos por distancia para evitar falsos positivos
                        if distance > 1000:  # Ajustar este umbral según sea necesario
                            far = tuple(contour[far_defect][0])
                            # Dibujar círculo azul en el defecto
                            cv2.circle(img_resultado, far, 7, [255, 0, 0], -1)
                            total_defectos += 1
                
                # Dibujar contornos
                cv2.drawContours(img_resultado, [orig_contour], -1, color=(0, 0, 0), thickness=2)
                cv2.drawContours(img_resultado, [contour], -1, color=(255, 0, 0), thickness=2)
                
        except Exception as e:
            # Si hay error en convexityDefects, continuar con el siguiente contorno
            continue
    
    return img_resultado, total_defectos

# === FUNCIONES PARA CAPÍTULO 8 - Detección de Color ===
def ejercicio_capitulo8(imagen, color_elegido="azul"):
    # Definir rangos de color en HSV
    rangos_color = {
        "azul": {
            "lower": np.array([100, 150, 0]),
            "upper": np.array([140, 255, 255])
        },
        "rojo": {
            "lower": np.array([0, 150, 0]),
            "upper": np.array([10, 255, 255])
        },
        "verde": {
            "lower": np.array([40, 150, 0]),
            "upper": np.array([80, 255, 255])
        },
        "amarillo": {
            "lower": np.array([20, 150, 0]),
            "upper": np.array([40, 255, 255])
        }
    }
    
    # Convertir a HSV
    hsv_frame = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # Obtener rangos del color elegido
    lower = rangos_color[color_elegido]["lower"]
    upper = rangos_color[color_elegido]["upper"]
    
    # Crear máscara
    mask = cv2.inRange(hsv_frame, lower, upper)
    
    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Bitwise-AND con la imagen original
    res = cv2.bitwise_and(imagen, imagen, mask=mask)
    res = cv2.medianBlur(res, ksize=5)
    
    # Contar píxeles del color detectado
    pixeles_color = cv2.countNonZero(mask)
    total_pixeles = imagen.shape[0] * imagen.shape[1]
    porcentaje_color = (pixeles_color / total_pixeles) * 100
    
    return res, mask, porcentaje_color

# === FUNCIONES PARA CAPÍTULO 10 - Cámara Simple ===
def ejercicio_capitulo10():
    st.info("📸 **Nota:** En Streamlit Cloud no se puede acceder a la cámara web directamente.")
    st.write("""
    **Tu código original del Capítulo 10:**
    ```python
    import cv2 
    import numpy as np
    
    # Define 'blue' range in HSV colorspace 
    lower = np.array([60,100,100]) 
    upper = np.array([180,255,255]) 
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        res = cv2.medianBlur(res, ksize=5)
        cv2.imshow('Original image', frame)
        cv2.imshow('Color Detector', res)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    ```
    """)
    
    st.warning("""
    **⚠️ Limitación en Streamlit Cloud:**
    - No se puede acceder a dispositivos de hardware como cámaras
    - El código original funciona localmente pero no en la nube
    - Usa el Capítulo 8 para detección de color en imágenes subidas
    """)

# === Sidebar para navegación ===
st.sidebar.title("🎯 Navegación")
capitulo = st.sidebar.selectbox(
    "Selecciona un capítulo:",
    [
        "🏠 Introducción", 
        "📷 Capítulo 1", "🌀 Capítulo 2", "🎨 Capítulo 3", "👤 Capítulo 4", "🔺 Capítulo 5",
        "✂️ Capítulo 6", "🔵 Capítulo 7", "🎨 Capítulo 8", "🐱 Capítulo 9", "📹 Capítulo 10", "💫 Capítulo 11"
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
    
    modo = st.radio("🎭 Selecciona el modo:", ["🖼️ Original", "🌈 Cartoon con Color", "✏️ Sketch (Sin Color)"])
    
    img = cargar_imagen()
    if img is not None:
        if modo == "🖼️ Original":
            resultado = img
        elif modo == "🌈 Cartoon con Color":
            resultado = cartoonize_image(img, ksize=5, sketch_mode=False)
        else:
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
        
        if num_faces > 0:
            st.success(f"✅ Se detectaron {num_faces} rostro(s) en la imagen")
        else:
            st.warning("⚠️ No se detectaron rostros en la imagen")

elif capitulo == "🔺 Capítulo 5":
    st.header("🔺 Capítulo 5: Detección de Esquinas")
    st.write("**Qué hace:** Detecta esquinas en imágenes usando el algoritmo Good Features to Track")
    
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
        
        if num_corners > 0:
            st.success(f"✅ Se detectaron {num_corners} esquina(s) en la imagen")
        else:
            st.warning("⚠️ No se detectaron esquinas en la imagen")

elif capitulo == "✂️ Capítulo 6":
    st.header("✂️ Capítulo 6: Seam Carving")
    st.write("**Qué hace:** Redimensionamiento inteligente que preserva el contenido importante eliminando o agregando 'costuras'")
    
    col_mode, col_seams = st.columns(2)
    with col_mode:
        modo = st.radio("Modo:", ["🗑️ Eliminar costuras", "➕ Agregar costuras"])
    with col_seams:
        num_seams = st.slider("Número de costuras:", 1, 100, 10, 1)
    
    img = cargar_imagen()
    if img is not None:
        with st.spinner("✂️ Procesando costuras..."):
            if modo == "🗑️ Eliminar costuras":
                resultado, costuras = ejercicio_capitulo6(img, num_seams, "eliminar")
                titulo_resultado = f"Imagen Reducida ({num_seams} costuras eliminadas)"
            else:
                resultado, costuras = ejercicio_capitulo6(img, num_seams, "agregar")
                titulo_resultado = f"Imagen Ampliada ({num_seams} costuras agregadas)"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, channels="BGR", caption="🖼️ Imagen Original")
        with col2:
            st.image(costuras, channels="BGR", caption="📍 Costuras Detectadas")
        with col3:
            st.image(resultado, channels="BGR", caption=titulo_resultado)

elif capitulo == "🔵 Capítulo 7":
    st.header("🔵 Capítulo 7: Defectos de Convexidad")
    st.write("**Qué hace:** Detecta puntos donde los contornos se alejan del casco convexo (útil para análisis de formas y gestos)")
    
    factor_epsilon = st.slider("Factor de aproximación:", 0.001, 0.1, 0.01, 0.001)
    
    img = cargar_imagen()
    if img is not None:
        with st.spinner("🔍 Analizando contornos y defectos..."):
            resultado, num_defectos = ejercicio_capitulo7(img, factor_epsilon)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="🖼️ Imagen Original")
        with col2:
            st.image(resultado, channels="BGR", caption=f"🔵 Defectos detectados: {num_defectos}")
        
        st.info("""
        **Leyenda:**
        - ⚫ **Contornos negros**: Contornos originales
        - 🔴 **Contornos rojos**: Contornos aproximados  
        - 🔵 **Círculos azules**: Defectos de convexidad
        """)
        
        if num_defectos > 0:
            st.success(f"✅ Se detectaron {num_defectos} defecto(s) de convexidad")
        else:
            st.warning("⚠️ No se detectaron defectos de convexidad")

elif capitulo == "🎨 Capítulo 8":
    st.header("🎨 Capítulo 8: Detección de Color")
    st.write("**Qué hace:** Detecta y aísla colores específicos en imágenes usando el espacio de color HSV")
    
    color_elegido = st.selectbox(
        "🎨 Selecciona el color a detectar:",
        ["azul", "rojo", "verde", "amarillo"]
    )
    
    img = cargar_imagen()
    if img is not None:
        with st.spinner(f"🔍 Detectando color {color_elegido}..."):
            resultado, mask, porcentaje = ejercicio_capitulo8(img, color_elegido)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, channels="BGR", caption="🖼️ Imagen Original")
        with col2:
            st.image(mask, caption="🎭 Máscara Binaria", use_column_width=True)
        with col3:
            st.image(resultado, channels="BGR", caption=f"🎨 Color {color_elegido} detectado")
        
        st.success(f"✅ **{porcentaje:.2f}%** de la imagen es de color **{color_elegido}**")

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

elif capitulo == "📹 Capítulo 10":
    st.header("📹 Capítulo 10: Detección de Color en Tiempo Real")
    st.write("**Qué hace:** Usa la cámara web para detectar color azul en tiempo real")
    ejercicio_capitulo10()

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
- ✅ Capítulo 6: Seam Carving
- ✅ Capítulo 7: Defectos de convexidad
- ✅ Capítulo 8: Detección de color
- ✅ Capítulo 9: Clasificación Perros/Gatos
- ✅ Capítulo 10: Explicación cámara
- ⏳ Capítulo 11: Pendiente
""")