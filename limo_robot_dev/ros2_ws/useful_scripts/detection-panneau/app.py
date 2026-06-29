import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np


# Remplace par le chemin ABSOLU de ton fichier best.pt

MODEL_PATH = r'C:\Users\ilyas\Documents\VS Code\Robot_Limo\Limo_Training\panneaux_run4\weights\best.pt'

# Chargement du modèle
print("Chargement du modèle en cours...")
model = YOLO(MODEL_PATH)
print("Modèle chargé !")

def detecter_image():
    # 1. Ouvrir l'explorateur de fichiers pour choisir une image
    filename = filedialog.askopenfilename(
        title="Choisir une image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not filename:
        return 

    # 2. Lancer la détection YOLO
    # conf=0.5 : on garde seulement si sur à 50%
    results = model.predict(source=filename, conf=0.5)
    
    # 3. Récupérer l'image résultat (avec les cadres dessinés)
    # YOLO renvoie l'image en format BGR (Bleu-Vert-Rouge)
    res_plotted = results[0].plot()
    
    # 4. Convertir l'image pour Tkinter (BGR -> RGB)
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(res_rgb)
    
    # 5. Redimensionner l'image si elle est trop grosse pour l'écran
    # On la limite à 800x600 max tout en gardant les proportions
    img_pil.thumbnail((800, 600))
    
    # Convertir en format compatible Tkinter
    img_tk = ImageTk.PhotoImage(img_pil)
    
    # 6. Mettre à jour l'interface
    label_image.config(image=img_tk)
    label_image.image = img_tk  # Indispensable pour éviter que l'image disparaisse (garbage collector)
    
    # Afficher le texte des détections
    noms_detectes = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]
        conf = float(box.conf[0])
        noms_detectes.append(f"{name} ({conf:.2f})")
        
    if noms_detectes:
        texte_resultat = "Détecté : " + ", ".join(noms_detectes)
        label_text.config(text=texte_resultat, fg="green")
    else:
        label_text.config(text="Aucun panneau détecté.", fg="red")

# --- CRÉATION DE LA FENÊTRE ---
fenetre = tk.Tk()
fenetre.title("Robot Limo - Détecteur de Panneaux")
fenetre.geometry("900x700") # Taille de la fenêtre
fenetre.config(bg="#f0f0f0") # Couleur de fond gris clair

# Titre
lbl_titre = Label(fenetre, text="Interface de Détection", font=("Arial", 20, "bold"), bg="#f0f0f0")
lbl_titre.pack(pady=10)

# Bouton pour charger
btn_load = Button(fenetre, text="Charger une image", command=detecter_image, font=("Arial", 14), bg="#007bff", fg="white")
btn_load.pack(pady=10)

# Zone de texte pour le résultat
label_text = Label(fenetre, text="En attente d'une image...", font=("Arial", 12), bg="#f0f0f0")
label_text.pack(pady=5)

# Zone pour afficher l'image
label_image = Label(fenetre, bg="#ddd") # Fond gris pour la zone image
label_image.pack(pady=10, padx=10, expand=True)

# Lancer l'application
fenetre.mainloop()