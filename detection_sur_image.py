import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import urllib.request
import os

# Téléchargement de l'image cameraman si elle n'existe pas
def get_cameraman_image():
    if not os.path.exists('cameraman.tif'):
        url = 'https://www.mathworks.com/matlabcentral/answers/uploaded_files/15876/cameraman.tif'
        try:
            urllib.request.urlretrieve(url, 'cameraman.tif')
            print("Image téléchargée avec succès.")
        except:
            print("Erreur lors du téléchargement de l'image. Veuillez vous assurer d'avoir une connexion internet.")
            # Créer une image de secours si le téléchargement échoue
            img = np.zeros((256, 256), dtype=np.uint8)
            cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
            cv2.imwrite('cameraman.tif', img)
            print("Image de secours créée.")

# Essayer de télécharger l'image
get_cameraman_image()

# Lecture de l'image cameraman
I = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

# Affichage de l'image originale
plt.figure()
plt.imshow(I, cmap='gray')
plt.title('Image originale (cameraman)')
plt.axis('off')

# Application des filtres gradients simples
h1 = np.array([[1, -1]])
I1 = cv2.filter2D(I, -1, h1, borderType=cv2.BORDER_REPLICATE)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(I1, cmap='gray')
plt.title('Filtre h1 = [1 -1]')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(I1), cmap='gray')
plt.title('abs(I1)')
plt.axis('off')

h2 = np.array([[1], [-1]])
I2 = cv2.filter2D(I, -1, h2, borderType=cv2.BORDER_REPLICATE)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(I2, cmap='gray')
plt.title('Filtre h2 = [1; -1]')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(I2), cmap='gray')
plt.title('abs(I2)')
plt.axis('off')

# Application des filtres de Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Equivalent to fspecial('sobel')' in MATLAB
I3 = cv2.filter2D(I, -1, sobel_x, borderType=cv2.BORDER_REPLICATE)
# Alternative: I3 = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(I3, cmap='gray')
plt.title('Filtre Sobel horizontal')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(I3), cmap='gray')
plt.title('abs(I3)')
plt.axis('off')

sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Equivalent to fspecial('sobel') in MATLAB
I5 = cv2.filter2D(I, -1, sobel_y, borderType=cv2.BORDER_REPLICATE)
# Alternative: I5 = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(I5, cmap='gray')
plt.title('Filtre Sobel vertical')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(I5), cmap='gray')
plt.title('abs(I5)')
plt.axis('off')

# Magnitude du gradient de Sobel
magnitude = np.sqrt(I3.astype(float)**2 + I5.astype(float)**2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude du gradient de Sobel')
plt.axis('off')

# Application du filtre Laplacien
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
I7 = cv2.filter2D(I, -1, laplacian_kernel, borderType=cv2.BORDER_REPLICATE)
# Alternative: I7 = cv2.Laplacian(I, cv2.CV_64F)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(I7, cmap='gray')
plt.title('Filtre Laplacien')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(I7), cmap='gray')
plt.title('abs(I7)')
plt.axis('off')

# Application du filtre LOG (Laplacian of Gaussian)
sigma = 1
# Créer une grille 2D centrée
x, y = np.meshgrid(np.arange(-7, 8), np.arange(-7, 8))
r_squared = x**2 + y**2
# Formule du LOG
log_filter = -(1/(np.pi*sigma**4)) * (1 - r_squared/(2*sigma**2)) * np.exp(-r_squared/(2*sigma**2))
# Normaliser le filtre
log_filter = log_filter - np.mean(log_filter)
I8 = cv2.filter2D(I, -1, log_filter, borderType=cv2.BORDER_REPLICATE)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(I8, cmap='gray')
plt.title(f'Filtre LOG, σ = {sigma}')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(I8), cmap='gray')
plt.title(f'abs(I8), σ = {sigma}')
plt.axis('off')

# Variation de sigma pour le filtre LOG
sigmas = [0.5, 1, 2, 4]
plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

for i, sigma in enumerate(sigmas):
    # Créer le filtre LOG
    x, y = np.meshgrid(np.arange(-7, 8), np.arange(-7, 8))
    r_squared = x**2 + y**2
    log_filter = -(1/(np.pi*sigma**4)) * (1 - r_squared/(2*sigma**2)) * np.exp(-r_squared/(2*sigma**2))
    log_filter = log_filter - np.mean(log_filter)
    I_log = cv2.filter2D(I, -1, log_filter, borderType=cv2.BORDER_REPLICATE)
    
    plt.subplot(2, 3, i+2)
    plt.imshow(np.abs(I_log), cmap='gray')
    plt.title(f'LOG, σ = {sigma}')
    plt.axis('off')

# Comparaison des différentes méthodes
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(np.abs(I1), cmap='gray')
plt.title('Gradient simple horizontal')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(np.abs(I2), cmap='gray')
plt.title('Gradient simple vertical')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(np.abs(I3), cmap='gray')
plt.title('Sobel horizontal')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(np.abs(I5), cmap='gray')
plt.title('Sobel vertical')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude du gradient de Sobel')
plt.axis('off')

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(I, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.abs(I7), cmap='gray')
plt.title('Laplacien')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(np.abs(I8), cmap='gray')
plt.title(f'LOG, σ = {sigma}')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.abs(I_log), cmap='gray')
plt.title(f'LOG, σ = {sigmas[-1]}')
plt.axis('off')

plt.tight_layout()
plt.show()