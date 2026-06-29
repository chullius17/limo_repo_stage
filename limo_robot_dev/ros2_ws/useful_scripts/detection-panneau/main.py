from ultralytics import YOLO
import cv2


model = YOLO(r'C:\Users\ilyas\Documents\VS Code\Robot_Limo\Limo_Training\panneaux_run1\weights\best.pt') 


results = model.predict(source=r'C:\Users\ilyas\Downloads\panneau-limitation-de-vitesse-80km-h-b1480.jpg', show=True, conf=0.5) 

#C:\Users\ilyas\Documents\VS Code\Robot_Limo\Robot_Limo2-1\test\images\00002_00028_jpg.rf.f0206dac51137cb649c444d054f0b853.jpg
#"C:\Users\ilyas\Downloads\panneau-limitation-de-vitesse-80km-h-b1480.jpg"
#Robot_Limo2-1\test\images\video2_frame_0096_filtered_jpg.rf.7154676a1e5b2e3893c33ca1a9394219.jpg
