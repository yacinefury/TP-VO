import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology
import cv2

# Charger l'image
circles = io.imread('circles.png', as_gray=True)

# Binariser l'image
circles_bin = circles > 0.5

# Calculer le squelette avec connectivité 4 (d4)
squelette_d4 = morphology.skeletonize(circles_bin)

# Calculer le squelette avec connectivité 8 (d8)
squelette_d8 = morphology.skeletonize(circles_bin, method='lee')

# Visualiser les résultats
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(circles, cmap='gray'), plt.title('Image originale')
plt.subplot(132), plt.imshow(squelette_d4, cmap='gray'), plt.title('Squelette d4')
plt.subplot(133), plt.imshow(squelette_d8, cmap='gray'), plt.title('Squelette d8')
plt.tight_layout()
plt.savefig('comparaison_squelettes.png')
plt.show()