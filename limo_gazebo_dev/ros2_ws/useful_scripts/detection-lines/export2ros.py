#!/usr/bin/env python3
import os
import shutil
from ultralytics import YOLO

# --- CONFIGURAZIONE PERCORSI ---
PT_MODEL_PATH = 'runs/segment/Limo_Lane_Segmentation/yolov8n_lane_run1/weights/best.pt'

# Percorso di destinazione all'interno del tuo workspace ROS 2
# Modificalo se il tuo workspace ha un percorso differente
DESTINATION_DIR = os.path.expanduser('~/limo_project/ros2_ws/src/cv_package/')
DESTINATION_FILE = os.path.join(DESTINATION_DIR, 'best.onnx')

def main():
    # 1. Verifica che il modello PyTorch esista
    if not os.path.exists(PT_MODEL_PATH):
        print(f"ERRORE: Non ho trovato il file PyTorch in: {PT_MODEL_PATH}")
        print("Controlla il percorso in cima allo script!")
        return

    print(f"Caricamento del modello PyTorch da: {PT_MODEL_PATH}...")
    model = YOLO(PT_MODEL_PATH)

    # 2. Esportazione in ONNX (Risoluzione bloccata a 448x448 per mantenere le ottimizzazioni)
    print("Avvio dell'esportazione in formato ONNX (imgsz=448)...")
    try:
        # opset=12 o 11 garantisce la massima compatibilità con ONNX Runtime su ROS
        onnx_path = model.export(format='onnx', imgsz=448, opset=12)
        print(f"Modello ONNX generato con successo in: {onnx_path}")
    except Exception as e:
        print(f"ERRORE durante l'esportazione: {str(e)}")
        return

    # 3. Spostamento del file dentro cv_package
    if not os.path.exists(DESTINATION_DIR):
        print(f"La cartella di destinazione ROS 2 non esiste, provo a crearla: {DESTINATION_DIR}")
        os.makedirs(DESTINATION_DIR, exist_ok=True)

    print(f"Spostamento del file ONNX in: {DESTINATION_FILE}...")
    try:
        # Se esiste già un vecchio best.onnx, viene sovrascritto in automatico
        shutil.move(onnx_path, DESTINATION_FILE)
        print("\n" + "="*50)
        print(" OPERAZIONE COMPLETATA CON SUCCESSO!")
        print(f" Il file è pronto dentro cv_package.")
        print("="*50)
    except Exception as e:
        print(f"ERRORE durante lo spostamento del file: {str(e)}")

if __name__ == '__main__':
    main()