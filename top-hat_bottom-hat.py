import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, filters
from scipy import ndimage
import cv2
from skimage.morphology import disk, square, opening, closing

# Charger l'image
rice = io.imread('rice.png', as_gray=True)

# Binariser l'image
rice_bin = rice > filters.threshold_otsu(rice)

# Appliquer top-hat blanc avec différentes tailles d'éléments structurants
se_small = square(5)
se_medium = square(12)
se_large = square(20)

tophat_small = morphology.white_tophat(rice, se_small)
tophat_medium = morphology.white_tophat(rice, se_medium)
tophat_large = morphology.white_tophat(rice, se_large)

# Binariser les résultats du top-hat
tophat_bin_small = tophat_small > filters.threshold_otsu(tophat_small)
tophat_bin_medium = tophat_medium > filters.threshold_otsu(tophat_medium)
tophat_bin_large = tophat_large > filters.threshold_otsu(tophat_large)

# Appliquer bottom-hat avec différentes tailles d'éléments structurants
bottomhat_small = morphology.black_tophat(rice, se_small)
bottomhat_medium = morphology.black_tophat(rice, se_medium)
bottomhat_large = morphology.black_tophat(rice, se_large)

# Binariser les résultats du bottom-hat
bottomhat_bin_small = bottomhat_small > filters.threshold_otsu(bottomhat_small)
bottomhat_bin_medium = bottomhat_medium > filters.threshold_otsu(bottomhat_medium)
bottomhat_bin_large = bottomhat_large > filters.threshold_otsu(bottomhat_large)

# Visualiser les résultats
plt.figure(figsize=(15, 12))
plt.subplot(341), plt.imshow(rice, cmap='gray'), plt.title('Image originale')
plt.subplot(342), plt.imshow(rice_bin, cmap='gray'), plt.title('Image binarisée')
plt.subplot(343), plt.imshow(tophat_small, cmap='gray'), plt.title('Top-hat (5x5)')
plt.subplot(344), plt.imshow(tophat_bin_small, cmap='gray'), plt.title('Top-hat bin (5x5)')
plt.subplot(345), plt.imshow(tophat_medium, cmap='gray'), plt.title('Top-hat (12x12)')
plt.subplot(346), plt.imshow(tophat_bin_medium, cmap='gray'), plt.title('Top-hat bin (12x12)')
plt.subplot(347), plt.imshow(tophat_large, cmap='gray'), plt.title('Top-hat (20x20)')
plt.subplot(348), plt.imshow(tophat_bin_large, cmap='gray'), plt.title('Top-hat bin (20x20)')
plt.tight_layout()
plt.savefig('tophat_comparaison.png')
plt.show()

plt.figure(figsize=(15, 12))
plt.subplot(341), plt.imshow(rice, cmap='gray'), plt.title('Image originale')
plt.subplot(342), plt.imshow(rice_bin, cmap='gray'), plt.title('Image binarisée')
plt.subplot(343), plt.imshow(bottomhat_small, cmap='gray'), plt.title('Bottom-hat (5x5)')
plt.subplot(344), plt.imshow(bottomhat_bin_small, cmap='gray'), plt.title('Bottom-hat bin (5x5)')
plt.subplot(345), plt.imshow(bottomhat_medium, cmap='gray'), plt.title('Bottom-hat (12x12)')
plt.subplot(346), plt.imshow(bottomhat_bin_medium, cmap='gray'), plt.title('Bottom-hat bin (12x12)')
plt.subplot(347), plt.imshow(bottomhat_large, cmap='gray'), plt.title('Bottom-hat (20x20)')
plt.subplot(348), plt.imshow(bottomhat_bin_large, cmap='gray'), plt.title('Bottom-hat bin (20x20)')
plt.tight_layout()
plt.savefig('bottomhat_comparaison.png')
plt.show()

# Traitement de l'image flou-test.png
flou = io.imread('flou-test.png', as_gray=True)

# Essayer différentes approches pour obtenir une bonne image binaire
# Approche 1: Top-hat suivi de binarisation
se_flou = disk(15)  # Un élément structurant plus grand pour les objets flous
tophat_flou = morphology.white_tophat(flou, se_flou)
tophat_bin_flou = tophat_flou > filters.threshold_otsu(tophat_flou)

# Approche 2: Amélioration du contraste puis binarisation
flou_eq = cv2.equalizeHist((flou*255).astype(np.uint8)) / 255.0
flou_eq_bin = flou_eq > filters.threshold_otsu(flou_eq)

# Approche 3: Bottom-hat suivi de binarisation inversée
bottomhat_flou = morphology.black_tophat(flou, se_flou)
bottomhat_bin_flou = bottomhat_flou > filters.threshold_otsu(bottomhat_flou)
bottomhat_bin_flou_inv = ~bottomhat_bin_flou

# Approche 4: Combinaison de top-hat et bottom-hat
combined = tophat_flou - bottomhat_flou
combined_bin = combined > filters.threshold_otsu(combined)

# Visualiser les résultats
plt.figure(figsize=(15, 12))
plt.subplot(331), plt.imshow(flou, cmap='gray'), plt.title('Image originale')
plt.subplot(332), plt.imshow(tophat_flou, cmap='gray'), plt.title('Top-hat')
plt.subplot(333), plt.imshow(tophat_bin_flou, cmap='gray'), plt.title('Top-hat binarisé')
plt.subplot(334), plt.imshow(flou_eq, cmap='gray'), plt.title('Égalisation d\'histogramme')
plt.subplot(335), plt.imshow(flou_eq_bin, cmap='gray'), plt.title('Égalisation binarisée')
plt.subplot(336), plt.imshow(bottomhat_flou, cmap='gray'), plt.title('Bottom-hat')
plt.subplot(337), plt.imshow(bottomhat_bin_flou_inv, cmap='gray'), plt.title('Bottom-hat inversé')
plt.subplot(338), plt.imshow(combined, cmap='gray'), plt.title('Combinaison')
plt.subplot(339), plt.imshow(combined_bin, cmap='gray'), plt.title('Combinaison binarisée')
plt.tight_layout()
plt.savefig('traitement_flou_test.png')
plt.show()