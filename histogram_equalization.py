import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lecture de l'image
img = cv2.imread('pont.pgm', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Erreur: Impossible de lire l'image")
    exit()

# Mesures de contraste pour l'image originale
min_val = np.min(img)
max_val = np.max(img)
michelson_orig = float(max_val - min_val) / float(max_val + min_val)

img_float = img.astype(float)
B_orig = np.mean(img_float)
rms_orig = np.sqrt(np.mean((img_float - B_orig) ** 2))

# Égalisation d'histogramme
img_egalisee = cv2.equalizeHist(img)

# Calcul des histogrammes
hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_egal = cv2.calcHist([img_egalisee], [0], None, [256], [0, 256])
x_values = np.arange(256)

# Mesures de contraste pour l'image égalisée
min_val_egal = np.min(img_egalisee)
max_val_egal = np.max(img_egalisee)
michelson_egal = float(max_val_egal - min_val_egal) / float(max_val_egal + min_val_egal)

img_egal_float = img_egalisee.astype(float)
B_egal = np.mean(img_egal_float)
rms_egal = np.sqrt(np.mean((img_egal_float - B_egal) ** 2))

# Affichage des résultats
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Image Originale')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_egalisee, cmap='gray')
plt.title('Image avec Égalisation d\'Histogramme')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.bar(x_values, hist_orig.flatten())
plt.title('Histogramme Original')
plt.xlim([0, 255])

plt.subplot(2, 2, 4)
plt.bar(x_values, hist_egal.flatten())
plt.title('Histogramme Égalisé')
plt.xlim([0, 255])

plt.tight_layout()

# Affichage des mesures de contraste
print(f'Image Originale - Contraste Michelson: {michelson_orig:.4f}, Contraste RMS: {rms_orig:.4f}')
print(f'Image Égalisée - Contraste Michelson: {michelson_egal:.4f}, Contraste RMS: {rms_egal:.4f}')

# Sauvegarde de l'image égalisée
cv2.imwrite('pont_egalise.png', img_egalisee)

# Implémentation manuelle de l'égalisation d'histogramme
h, w = img.shape
N = h * w  # Nombre total de pixels
hist_cumul = np.cumsum(hist_orig)  # Histogramme cumulé
T = np.uint8((255 / N) * hist_cumul)  # Fonction de transformation

# Application de la fonction de transformation
img_egal_manuelle = np.zeros_like(img, dtype=np.uint8)
for i in range(256):
    img_egal_manuelle[img == i] = T[i]

# Vérification de l'implémentation manuelle
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_egalisee, cmap='gray')
plt.title('Égalisation avec cv2.equalizeHist()')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_egal_manuelle, cmap='gray')
plt.title('Égalisation manuelle')
plt.axis('off')

plt.tight_layout()
plt.show()