from ultralytics import YOLO

# 1. Charger ton modèle 
model_path = r'C:\Users\ilyas\Documents\VS Code\Robot_Limo\Limo_Training\panneaux_run4\weights\best.pt'
model = YOLO(model_path)

# 2. Chemin de ta vidéo
video_path = r'C:\Users\ilyas\Downloads\video_test3_mp.mp4'

# 3. Lancer la détection
# save=True va créer une copie de la vidéo avec les rectangles dessinés
# conf=0.5 signifie qu'on ne garde que les détections sûres à 50%
print("Traitement de la vidéo en cours...")
results = model.predict(source=video_path, save=True, show=True, conf=0.5)

print("Terminé !")