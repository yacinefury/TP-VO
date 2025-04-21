import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 1. Création de l'image avec carré
I = np.ones((250, 250)) * 50  # Fond gris foncé (50)
I[100:150, 100:150] = 200  # Carré gris clair (200) au milieu

plt.figure()
plt.imshow(I.astype(np.uint8), cmap='gray')
plt.title('Image originale (carré)')
plt.axis('off')

# 2. Filtres gradients simples
# Filtre gradient horizontal
h1 = np.array([[1, -1]])
I1 = cv2.filter2D(I, -1, h1, borderType=cv2.BORDER_REPLICATE)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I.astype(np.uint8), cmap='gray')
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

# Filtre gradient vertical
h2 = np.array([[1], [-1]])
I2 = cv2.filter2D(I, -1, h2, borderType=cv2.BORDER_REPLICATE)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I.astype(np.uint8), cmap='gray')
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

# 3. Filtres de Sobel
# Sobel horizontal
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Equivalent to fspecial('sobel')' in MATLAB
I3 = cv2.filter2D(I, -1, sobel_x, borderType=cv2.BORDER_REPLICATE)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I.astype(np.uint8), cmap='gray')
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

# Sobel vertical
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Equivalent to fspecial('sobel') in MATLAB
I5 = cv2.filter2D(I, -1, sobel_y, borderType=cv2.BORDER_REPLICATE)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I.astype(np.uint8), cmap='gray')
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
I4 = I3  # h4 = h3 in the MATLAB code
I6 = I5  # h6 = h5 in the MATLAB code
magnitude = np.sqrt(I4**2 + I6**2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(I.astype(np.uint8), cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude du gradient de Sobel')
plt.axis('off')

# 4. Filtre Laplacien
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Equivalent to fspecial('laplacian', 0) in MATLAB
I7 = cv2.filter2D(I, -1, laplacian_kernel, borderType=cv2.BORDER_REPLICATE)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(I.astype(np.uint8), cmap='gray')
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

# 5. Filtre LOG (Laplacian of Gaussian)
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
plt.imshow(I.astype(np.uint8), cmap='gray')
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
plt.imshow(I.astype(np.uint8), cmap='gray')
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

plt.tight_layout()
plt.show()