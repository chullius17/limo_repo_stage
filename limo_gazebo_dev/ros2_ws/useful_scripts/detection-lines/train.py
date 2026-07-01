from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Charger un modèle pré-entraîné
    # On utilise 'yolov8n.pt' (Nano) car c'est le plus léger et le plus rapide 
    # pour un robot comme le Limo.
    model = YOLO('yolov8n-seg.pt')

    # 2. Lancer l'entraînement
    # data : chemin vers le fichier data.yaml téléchargé par Roboflow
    # epochs : nombre de fois que le modèle voit toutes les images (50 est un bon début)
    # imgsz : taille des images (640 est standard)
    # device : 0 pour utiliser ta carte graphique GPU
    model.train(
        data='data.yaml',  
        epochs=100, 
        imgsz=640, 
        device=0,
        batch=8,      # <── RIDUCIAMO DA 16 A 8 (dimezza la RAM consumata)
        workers=2,    # <── RIDUCIAMO DA 8 A 2 (riduce drasticamente lo sforzo della CPU)
        project='Limo_Lane_Segmentation',
        name='yolov8n_lane_run1',
        resume=True,   # <── AGGIUNGI QUESTO!
        patience=15
    )

