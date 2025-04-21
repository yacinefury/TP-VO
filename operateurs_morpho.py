import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fonction pour afficher plusieurs images
def display_images(images, titles, rows, cols):
    plt.figure(figsize=(15, 10))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Chargement de l'image
try:
    image = cv2.imread('formes.png', cv2.IMREAD_GRAYSCALE)
    # Si l'image n'existe pas, on crée une image synthétique avec des formes
    if image is None:
        print("Image 'formes.png' non trouvée, création d'une image synthétique...")
        image = np.zeros((400, 400), dtype=np.uint8)
        # Dessiner des rectangles
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
        cv2.rectangle(image, (200, 200), (350, 350), 255, -1)
        # Dessiner des cercles
        cv2.circle(image, (300, 100), 50, 255, -1)
        cv2.circle(image, (100, 300), 40, 255, -1)
        # Ajouter du bruit
        noise = np.random.randint(0, 2, size=image.shape, dtype=np.uint8) * 255
        noise_mask = np.random.random(size=image.shape) > 0.95
        image[noise_mask] = noise[noise_mask]
except Exception as e:
    print(f"Erreur lors du chargement de l'image: {e}")
    # Créer une image synthétique en cas d'erreur
    image = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
    cv2.rectangle(image, (200, 200), (350, 350), 255, -1)
    cv2.circle(image, (300, 100), 50, 255, -1)
    cv2.circle(image, (100, 300), 40, 255, -1)
    noise = np.random.randint(0, 2, size=image.shape, dtype=np.uint8) * 255
    noise_mask = np.random.random(size=image.shape) > 0.95
    image[noise_mask] = noise[noise_mask]

# Définir les noyaux pour les opérations morphologiques
kernel_small = np.ones((3, 3), np.uint8)
kernel_medium = np.ones((5, 5), np.uint8)
kernel_large = np.ones((7, 7), np.uint8)

# Résultat 1: Éliminer le bruit tout en préservant les formes principales
# Opération d'ouverture (érosion suivie de dilatation)
result1 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_small)

# Résultat 2: Séparer les formes qui se touchent
# Érosion pour séparer les formes
result2 = cv2.erode(image, kernel_medium, iterations=2)

# Résultat 3: Combler les trous dans les formes
# Fermeture (dilatation suivie d'érosion)
result3 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_medium)

# Afficher les résultats
display_images([image, result1, result2, result3], 
               ['Image originale', 
                'Résultat 1: Élimination du bruit (Ouverture)', 
                'Résultat 2: Séparation des formes (Érosion)', 
                'Résultat 3: Comblement des trous (Fermeture)'], 
               2, 2)

# Bonus: Combinaison d'opérateurs pour un traitement plus avancé
# Extraction des contours
contours = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel_small)

# Squelettisation (approximation)
def skeletonize(img):
    skeleton = np.zeros(img.shape, np.uint8)
    img_copy = img.copy()
    while True:
        # Érosion
        eroded = cv2.erode(img_copy, kernel_small)
        # Ouverture après érosion
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel_small)
        # Soustraction
        diff = cv2.subtract(eroded, opened)
        # Union des points du squelette
        skeleton = cv2.bitwise_or(skeleton, diff)
        # Itération suivante
        img_copy = eroded.copy()
        if cv2.countNonZero(img_copy) == 0:
            break
    return skeleton

# Tentative de squelettisation (peut prendre du temps)
try:
    skeleton = skeletonize(image)
    
    # Afficher les résultats bonus
    display_images([image, contours, skeleton], 
                   ['Image originale', 
                    'Extraction des contours (Gradient)', 
                    'Squelettisation'], 
                   1, 3)
except Exception as e:
    print(f"Erreur lors de la squelettisation: {e}")
    # Afficher seulement les contours si la squelettisation échoue
    display_images([image, contours], 
                   ['Image originale', 
                    'Extraction des contours (Gradient)'], 
                   1, 2)

print("Explication des opérations morphologiques utilisées:")
print("1. Ouverture: Élimine le bruit tout en préservant les structures principales")
print("2. Érosion: Réduit la taille des objets, utile pour séparer les formes connectées")
print("3. Fermeture: Comble les petits trous et fissures dans les objets")
print("4. Gradient: Extrait les contours en calculant la différence entre la dilatation et l'érosion")
print("5. Squelettisation: Réduit les formes à leurs squelettes d'un pixel de large")
