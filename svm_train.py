import os
import pickle
from create_features import load_input_map, FeatureExtractor, extract_feature_map
from training import ClassifierTrainer

# === CONFIGURACIÓN ===
DATASET_DIR = "dataset"          # Carpeta donde están las imágenes
MODELS_DIR = "models"            # Carpeta donde guardaremos los modelos
CODEBOOK_FILE = os.path.join(MODELS_DIR, "codebook.pkl")
FEATURE_MAP_FILE = os.path.join(MODELS_DIR, "feature_map.pkl")
SVM_FILE = os.path.join(MODELS_DIR, "svm_model.pkl")

# === CREAR CARPETA MODELS SI NO EXISTE ===
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# === CARGAR LAS IMÁGENES ===
print("🔍 Cargando imágenes del dataset...")

# Asegúrate de que tus carpetas se llamen 'perros' y 'gatos'
input_map = []
input_map += load_input_map("perro", os.path.join(DATASET_DIR, "perros"))
input_map += load_input_map("gato", os.path.join(DATASET_DIR, "gatos"))

# === CREAR EL CODEBOOK (KMeans + centroides) ===
print("\n📘 Construyendo el codebook (esto puede tardar un poco)...")
kmeans, centroids = FeatureExtractor().get_centroids(input_map)

# 🔧 Convertir centroides del modelo a float32 para evitar errores de tipo
kmeans.cluster_centers_ = kmeans.cluster_centers_.astype("float32")
centroids = kmeans.cluster_centers_

with open(CODEBOOK_FILE, "wb") as f:
    pickle.dump((kmeans, centroids), f)
print("✅ Codebook guardado en", CODEBOOK_FILE)

# === CREAR EL FEATURE MAP ===
print("\n⚙️ Extrayendo el mapa de características...")
feature_map = extract_feature_map(input_map, kmeans, centroids)

with open(FEATURE_MAP_FILE, "wb") as f:
    pickle.dump(feature_map, f)
print("✅ Feature map guardado en", FEATURE_MAP_FILE)

# === ENTRENAR EL MODELO SVM ===
print("\n🤖 Entrenando el modelo SVM...")
labels_words = [x['label'] for x in feature_map]
dim_size = feature_map[0]['feature_vector'].shape[1]
X = [x['feature_vector'].reshape(dim_size,) for x in feature_map]

svm = ClassifierTrainer(X, labels_words)

with open(SVM_FILE, "wb") as f:
    pickle.dump(svm, f)
print("✅ Modelo SVM guardado en", SVM_FILE)

print("\n🎉 ¡Entrenamiento completado exitosamente!")
