import cv2
import os
import math
from pathlib import Path

def video_to_frames(video_path, output_folder, starting_index=0):
    # Vérifier si le fichier existe
    if not os.path.exists(video_path):
        print(f"Erreur : La vidéo '{video_path}' est introuvable.")
        return

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Dossier '{output_folder}' créé.")

    # Capturer la vidéo
    cap = cv2.VideoCapture(str(video_path))
    
    # Récupérer le nombre d'images par seconde (FPS) de la vidéo originale
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # On arrondit car le FPS peut être un nombre flottant (ex: 29.97)
    # Si la vidéo est à 30fps, on veut sauvegarder chaque 30ème image pour avoir 1 image/sec
    #frame_interval = math.ceil(original_fps)
    frame_interval = 8
    
    print(f"FPS de la vidéo : {original_fps}")
    print(f"Extraction d'une image toutes les {frame_interval} frames...")

    frame_count = 0
    saved_count = 0

    while True:
        # Lire une frame
        success, frame = cap.read()

        # Si 'success' est faux, c'est la fin de la vidéo
        if not success:
            break

        # Si le numéro de la frame actuelle est un multiple de l'intervalle (FPS)
        # On sauvegarde l'image
        if frame_count % frame_interval == 0:
            # Nom du fichier : frame_0.jpg, frame_1.jpg, etc.
            filename = os.path.join(output_folder, f"image_{starting_index + saved_count}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        
        frame_count += 1

    # Libérer la ressource vidéo
    cap.release()
    print(f"Terminé ! {saved_count} images ont été extraites dans '{output_folder}'.")

# --- Configuration ---
current_dir = Path(__file__).parent

video_input_gazebo = current_dir / "Training_material" / "Training_gazebo.webm"
video_input_irl = current_dir / "Training_material" / "Training_irl.mp4"
dossier_sortie = current_dir / "Training_material" / "images_extracted"

# Lancer la fonction
video_to_frames(str(video_input_gazebo), str(dossier_sortie))
video_to_frames(str(video_input_irl), str(dossier_sortie), starting_index=1000)