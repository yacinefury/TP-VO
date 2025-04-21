import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.util import img_as_ubyte, img_as_bool
import urllib.request
import os

# Téléchargement de l'image text.png si elle n'existe pas
def get_text_image():
    if not os.path.exists('text.png'):
        url = 'https://raw.githubusercontent.com/scikit-image/scikit-image/master/skimage/data/text.png'
        try:
            urllib.request.urlretrieve(url, 'text.png')
            print("Image téléchargée avec succès.")
        except:
            print("Erreur lors du téléchargement de l'image. Veuillez vous assurer d'avoir une connexion internet.")
            # Créer une image de texte de secours si le téléchargement échoue
            img = np.ones((200, 400), dtype=np.uint8) * 255
            cv2.putText(img, "Exemple de texte", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            cv2.imwrite('text.png', img)
            print("Image de secours créée.")

# Essayer de télécharger l'image
get_text_image()

# Lecture de l'image de texte
text_img = cv2.imread('text.png', cv2.IMREAD_GRAYSCALE)
if text_img is None:
    print("Erreur: Impossible de lire l'image")
    exit()

# Conversion en image binaire si nécessaire
_, text_img = cv2.threshold(text_img, 127, 255, cv2.THRESH_BINARY)
# Inverser l'image pour avoir le texte en blanc (1) et le fond en noir (0)
text_img = 255 - text_img
# Convertir en booléen (True pour le texte, False pour le fond)
text_img_bool = text_img > 0

plt.figure()
plt.imshow(text_img_bool, cmap='gray')
plt.title('Image originale (text.png)')
plt.axis('off')

# Définition des éléments structurants
# Ligne horizontale
se_line_h = morphology.rectangle(1, 9)
# Ligne verticale
se_line_v = morphology.rectangle(9, 1)
# Carré 3x3
se_square3 = morphology.square(3)
# Carré 5x5
se_square5 = morphology.square(5)
# Disque de rayon 3
se_disk3 = morphology.disk(3)
# Disque de rayon 5
se_disk5 = morphology.disk(5)

# Dilatation avec différents éléments structurants
dilate_line_h = morphology.dilation(text_img_bool, se_line_h)
dilate_line_v = morphology.dilation(text_img_bool, se_line_v)
dilate_square3 = morphology.dilation(text_img_bool, se_square3)
dilate_square5 = morphology.dilation(text_img_bool, se_square5)
dilate_disk3 = morphology.dilation(text_img_bool, se_disk3)
dilate_disk5 = morphology.dilation(text_img_bool, se_disk5)

# Affichage des résultats de dilatation
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(dilate_line_h, cmap='gray')
plt.title('Dilatation - Ligne horizontale 9px')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(dilate_line_v, cmap='gray')
plt.title('Dilatation - Ligne verticale 9px')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dilate_square3, cmap='gray')
plt.title('Dilatation - Carré 3x3')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(dilate_square5, cmap='gray')
plt.title('Dilatation - Carré 5x5')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(dilate_disk3, cmap='gray')
plt.title('Dilatation - Disque r=3')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(dilate_disk5, cmap='gray')
plt.title('Dilatation - Disque r=5')
plt.axis('off')

# Érosion avec différents éléments structurants
erode_line_h = morphology.erosion(text_img_bool, se_line_h)
erode_line_v = morphology.erosion(text_img_bool, se_line_v)
erode_square3 = morphology.erosion(text_img_bool, se_square3)
erode_square5 = morphology.erosion(text_img_bool, se_square5)
erode_disk3 = morphology.erosion(text_img_bool, se_disk3)
erode_disk5 = morphology.erosion(text_img_bool, se_disk5)

# Affichage des résultats d'érosion
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(erode_line_h, cmap='gray')
plt.title('Érosion - Ligne horizontale 9px')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(erode_line_v, cmap='gray')
plt.title('Érosion - Ligne verticale 9px')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(erode_square3, cmap='gray')
plt.title('Érosion - Carré 3x3')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(erode_square5, cmap='gray')
plt.title('Érosion - Carré 5x5')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(erode_disk3, cmap='gray')
plt.title('Érosion - Disque r=3')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(erode_disk5, cmap='gray')
plt.title('Érosion - Disque r=5')
plt.axis('off')

# Vérification de la dualité Dilatation-Érosion
# Selon la dualité : ~(I ⊕ B) = (~I) ⊖ B
dual_test1 = ~dilate_square3
dual_test2 = morphology.erosion(~text_img_bool, se_square3)
diff_dual = np.logical_xor(dual_test1, dual_test2)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(dual_test1, cmap='gray')
plt.title('~(I ⊕ B)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(dual_test2, cmap='gray')
plt.title('(~I) ⊖ B')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(diff_dual, cmap='gray')
plt.title('Différence (doit être nulle)')
plt.axis('off')

# Ouverture et fermeture avec différentes tailles et formes d'ES
open_square3 = morphology.opening(text_img_bool, se_square3)
open_square5 = morphology.opening(text_img_bool, se_square5)
open_disk3 = morphology.opening(text_img_bool, se_disk3)
open_disk5 = morphology.opening(text_img_bool, se_disk5)

close_square3 = morphology.closing(text_img_bool, se_square3)
close_square5 = morphology.closing(text_img_bool, se_square5)
close_disk3 = morphology.closing(text_img_bool, se_disk3)
close_disk5 = morphology.closing(text_img_bool, se_disk5)

# Affichage des résultats d'ouverture
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(text_img_bool, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(open_square3, cmap='gray')
plt.title('Ouverture - Carré 3x3')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(open_square5, cmap='gray')
plt.title('Ouverture - Carré 5x5')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(open_disk3, cmap='gray')
plt.title('Ouverture - Disque r=3')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(open_disk5, cmap='gray')
plt.title('Ouverture - Disque r=5')
plt.axis('off')

# Affichage des résultats de fermeture
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(text_img_bool, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(close_square3, cmap='gray')
plt.title('Fermeture - Carré 3x3')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(close_square5, cmap='gray')
plt.title('Fermeture - Carré 5x5')
plt.axis('off')

plt.subplot