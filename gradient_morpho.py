import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from scipy import ndimage
import cv2
from skimage.morphology import disk, square, diamond, erosion, dilation

# Fonction pour calculer le gradient morphologique
def gradient_morphologique(image, element_structurant):
    dilate = dilation(image, element_structurant)
    erode = erosion(image, element_structurant)
    return dilate - erode

# Charger les images
circles = io.imread('circles.png', as_gray=True)
cameraman = io.imread('cameraman.tif', as_gray=True)

# Définir les différents éléments structurants
carre3 = square(3)
carre7 = square(7)
diam1 = diamond(1)
diam5 = diamond(5)

# Appliquer le gradient morphologique avec différents éléments structurants sur circles.png
grad_carre3_circles = gradient_morphologique(circles, carre3)
grad_carre7_circles = gradient_morphologique(circles, carre7)
grad_diam1_circles = gradient_morphologique(circles, diam1)
grad_diam5_circles = gradient_morphologique(circles, diam5)

# Appliquer les gradients numériques classiques sur circles.png
sobel_circles = filters.sobel(circles)
prewitt_circles = filters.prewitt(circles)
roberts_circles = filters.roberts(circles)

# Visualiser les résultats pour circles.png
plt.figure(figsize=(15, 10))
plt.subplot(241), plt.imshow(circles, cmap='gray'), plt.title('Image originale')
plt.subplot(242), plt.imshow(grad_carre3_circles, cmap='gray'), plt.title('Gradient morpho - carré 3')
plt.subplot(243), plt.imshow(grad_carre7_circles, cmap='gray'), plt.title('Gradient morpho - carré 7')
plt.subplot(244), plt.imshow(grad_diam1_circles, cmap='gray'), plt.title('Gradient morpho - diamond 1')
plt.subplot(245), plt.imshow(grad_diam5_circles, cmap='gray'), plt.title('Gradient morpho - diamond 5')
plt.subplot(246), plt.imshow(sobel_circles, cmap='gray'), plt.title('Gradient de Sobel')
plt.subplot(247), plt.imshow(prewitt_circles, cmap='gray'), plt.title('Gradient de Prewitt')
plt.subplot(248), plt.imshow(roberts_circles, cmap='gray'), plt.title('Gradient de Roberts')
plt.tight_layout()
plt.savefig('gradient_comparaison_circles.png')
plt.show()

# Répéter pour cameraman.tif
grad_carre3_cam = gradient_morphologique(cameraman, carre3)
grad_carre7_cam = gradient_morphologique(cameraman, carre7)
grad_diam1_cam = gradient_morphologique(cameraman, diam1)
grad_diam5_cam = gradient_morphologique(cameraman, diam5)

sobel_cam = filters.sobel(cameraman)
prewitt_cam = filters.prewitt(cameraman)
roberts_cam = filters.roberts(cameraman)

plt.figure(figsize=(15, 10))
plt.subplot(241), plt.imshow(cameraman, cmap='gray'), plt.title('Image originale')
plt.subplot(242), plt.imshow(grad_carre3_cam, cmap='gray'), plt.title('Gradient morpho - carré 3')
plt.subplot(243), plt.imshow(grad_carre7_cam, cmap='gray'), plt.title('Gradient morpho - carré 7')
plt.subplot(244), plt.imshow(grad_diam1_cam, cmap='gray'), plt.title('Gradient morpho - diamond 1')
plt.subplot(245), plt.imshow(grad_diam5_cam, cmap='gray'), plt.title('Gradient morpho - diamond 5')
plt.subplot(246), plt.imshow(sobel_cam, cmap='gray'), plt.title('Gradient de Sobel')
plt.subplot(247), plt.imshow(prewitt_cam, cmap='gray'), plt.title('Gradient de Prewitt')
plt.subplot(248), plt.imshow(roberts_cam, cmap='gray'), plt.title('Gradient de Roberts')
plt.tight_layout()
plt.savefig('gradient_comparaison_cameraman.png')
plt.show()