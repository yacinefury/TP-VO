import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, measure, segmentation, color
from scipy import ndimage
import cv2

# Charger l'image
text = io.imread('text.png', as_gray=True)

# Binariser l'image (inverser car le texte est noir sur fond blanc)
text_bin = text < 0.5

# Détecter les lettres horizontalement et verticalement
# 1. Isoler les caractères individuels
labels = measure.label(text_bin)
props = measure.regionprops(labels)

# Créer une image pour les lettres 'e' potentielles
e_candidates = np.zeros_like(text_bin, dtype=bool)

# Caractéristiques approximatives des lettres 'e'
for prop in props:
    # Calculer le ratio largeur/hauteur
    y0, x0, y1, x1 = prop.bbox
    height = y1 - y0
    width = x1 - x0
    ratio = width / height
    area = prop.area
    
    # Calculer le nombre de trous (caractéristique typique du 'e')
    region = prop.image
    holes = 1 - measure.label(~region)
    num_holes = len(np.unique(holes)) - 2  # -2 car 0 est le fond et -1 est l'objet lui-même
    
    # Vérifier si c'est potentiellement un 'e'
    if 0.7 < ratio < 1.3 and num_holes == 1:
        e_candidates[y0:y1, x0:x1] = region

# Opérations morphologiques pour raffiner la détection
se = morphology.disk(1)
e_refined = morphology.binary_opening(e_candidates, se)

# Étiqueter les lettres 'e' détectées
e_labels = measure.label(e_refined)
e_props = measure.regionprops(e_labels)
num_e = len(e_props)

# Créer une image colorée pour visualiser les résultats
text_rgb = color.gray2rgb(text)
for prop in e_props:
    y0, x0, y1, x1 = prop.bbox
    cv2.rectangle(text_rgb, (x0, y0), (x1, y1), (1, 0, 0), 2)

# Visualiser les résultats
plt.figure(figsize=(12, 8))
plt.subplot(221), plt.imshow(text, cmap='gray'), plt.title('Image originale')
plt.subplot(222), plt.imshow(text_bin, cmap='gray'), plt.title('Image binarisée')
plt.subplot(223), plt.imshow(e_candidates, cmap='gray'), plt.title('Candidats lettres "e"')
plt.subplot(224), plt.imshow(text_rgb), plt.title(f'Détection finale: {num_e} lettres "e"')
plt.tight_layout()
plt.savefig('detection_lettres_e.png')
plt.show()