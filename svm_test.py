import cv2
import pickle
import numpy as np
from create_features import FeatureExtractor

# === RUTAS DE LOS MODELOS ===
CODEBOOK_FILE = "models/codebook.pkl"
SVM_FILE = "models/svm_model.pkl"

# === CARGAR MODELOS ===
print("🔍 Cargando modelos entrenados...")
with open(CODEBOOK_FILE, "rb") as f:
    kmeans, centroids = pickle.load(f)

with open(SVM_FILE, "rb") as f:
    svm = pickle.load(f)

print("✅ Modelos cargados correctamente")

# === CLASIFICADOR DE IMÁGENES ===
class ImageClassifier:
    def __init__(self, svm, kmeans, centroids):
        self.svm = svm
        self.kmeans = kmeans
        self.centroids = centroids
        self.extractor = FeatureExtractor()

    def predict(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print("⚠️ Error: no se pudo cargar la imagen.")
            return None

        fv = self.extractor.get_feature_vector(img, self.kmeans, self.centroids)

        prediction = self.svm.clf.predict(fv)
        return prediction[0]

# === USO ===
if __name__ == "__main__":
    # 🔹 Ruta de prueba: cambia la imagen según quieras probar
    test_image = "test_imagenes/prueba1.jpg"
    # test_image = "dataset/gatos/gato1.jpg"

    classifier = ImageClassifier(svm, kmeans, centroids)
    result = classifier.predict(test_image)

    labels = {0: "Gato", 1: "Perro"}

    nombre = labels.get(result, "Desconocido")

    print(f"📸 Resultado para '{test_image}': {nombre}")
