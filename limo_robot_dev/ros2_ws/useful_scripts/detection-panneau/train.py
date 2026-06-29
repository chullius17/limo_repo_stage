from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Charger un modèle pré-entraîné
    # On utilise 'yolov8n.pt' (Nano) car c'est le plus léger et le plus rapide 
    # pour un robot comme le Limo.
    model = YOLO(r'Limo_Training\panneaux_run3\weights\best.pt')

    # 2. Lancer l'entraînement
    # data : chemin vers le fichier data.yaml téléchargé par Roboflow
    # epochs : nombre de fois que le modèle voit toutes les images (50 est un bon début)
    # imgsz : taille des images (640 est standard)
    # device : 0 pour utiliser ta carte graphique GPU
    model.train(
        data='Robot_Limo.v4-2/data.yaml',  
        epochs=50, 
        imgsz=640, 
        device=0,
        project='Limo_Training',
        name='panneaux_run4'
    )

