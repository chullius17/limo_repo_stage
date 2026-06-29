import torch
import time

# Vérifier si CUDA est dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Exécution sur : {device}")

if device.type == 'cuda':
    print(f"Carte : {torch.cuda.get_device_name(0)}")
    
    # Création de deux matrices aléatoires sur le GPU
    # (Taille 10000x10000 pour faire chauffer un peu la carte)
    x = torch.randn(10000, 10000, device=device)
    y = torch.randn(10000, 10000, device=device)

    print("Début du calcul matriciel...")
    start = time.time()
    # Multiplication matricielle
    z = torch.matmul(x, y)
    end = time.time()
    
    print(f"Calcul terminé en {end - start:.4f} secondes !")
else:
    print("Pas de GPU détecté.")