import cv2
import numpy as np
import face_recognition
import os
import streamlit as st

# Définir la couleur de fond et du texte


# Chemin du dossier contenant les images similaires
SIMILAR_IMAGES_FOLDER = '.\dataImg'

# Chargement de la base de signatures
signatures_db = np.load('FaceSignature_db2.npy', allow_pickle=True)

# Extraction des caractéristiques faciales uniquement
signatures_encodings = signatures_db[:, :-1].astype(np.float64)

# Fonction pour trouver les visages similaires avec un seuil de distance ajusté
def find_similar_faces(target_image, threshold=0.5):
    """Trouve les visages similaires dans la base de données."""
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    target_encoding = face_recognition.face_encodings(target_image)
    if not target_encoding:
        return []

    target_encoding = target_encoding[0]

    distances = face_recognition.face_distance(signatures_encodings, target_encoding)

    similar_faces = [(signatures_db[i][-1], distances[i]) for i in range(len(distances)) if distances[i] <= threshold]

    return similar_faces

# Fonction pour obtenir les chemins d'accès aux images similaires
def get_similar_images_paths(similar_faces):
    """Obtient les chemins d'accès aux images similaires."""
    similar_images_paths = []
    for name, _ in similar_faces:
        image_path = os.path.join(SIMILAR_IMAGES_FOLDER, name + '.jpg')  # Supposant que les images sont en format jpg
        similar_images_paths.append(image_path)
    return similar_images_paths

# Interface utilisateur Streamlit
st.title("Recherche de Visages Similaires")

uploaded_file = st.file_uploader("Charger une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affichage de l'image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(img, caption='Image chargée', width=400)
    
    # Recherche de visages similaires
    similar_faces = find_similar_faces(img)
    
    # Affichage des résultats
    if similar_faces:
        st.header("Visages similaires trouvés :")
        similar_images_paths = get_similar_images_paths(similar_faces)
        for image_path in similar_images_paths:
            # Vérifier si l'image chargée est la même que l'image similaire actuelle
            if os.path.basename(image_path) != uploaded_file.name:
                loaded_image = cv2.imread(image_path)
                if loaded_image is not None:
                    st.image(loaded_image, caption='Image similaire', width=200)  # Ajustez la largeur ici
                else:
                    st.write(f"Impossible de charger l'image {image_path}")
    else:
        st.write("Aucun visage similaire trouvé dans la base de données.")
