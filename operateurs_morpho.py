import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.util import img_as_ubyte, img_as_bool
import os

def display_images(images, titles, rows, cols, figsize=(15, 10)):
    """Function to display multiple images with titles"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Load the image 'formes.png' or create a synthetic one if not available
try:
    image = cv2.imread('formes.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image 'formes.png' not found")
    
    # Threshold the image if it's not already binary
    if np.max(image) > 1:
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Convert to boolean format (True for shapes, False for background)
    image_bool = image > 0
    
except FileNotFoundError:
    print("Creating synthetic 'formes.png' since the original wasn't found...")
    # Create a synthetic image with various shapes
    image = np.zeros((400, 400), dtype=np.uint8)
    
    # Add rectangles
    cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
    cv2.rectangle(image, (220, 220), (350, 350), 255, -1)
    
    # Add circles
    cv2.circle(image, (300, 100), 50, 255, -1)
    cv2.circle(image, (100, 300), 40, 255, -1)
    
    # Add some noise
    noise = np.random.randint(0, 2, size=image.shape, dtype=np.uint8) * 255
    noise_mask = np.random.random(size=image.shape) > 0.95
    image[noise_mask] = noise[noise_mask]
    
    # Add some holes inside shapes
    cv2.circle(image, (100, 100), 20, 0, -1)
    cv2.circle(image, (300, 100), 20, 0, -1)
    
    # Add some connections between shapes
    cv2.line(image, (150, 100), (220, 100), 255, 5)
    
    # Convert to boolean format
    image_bool = image > 0

# Define various structuring elements
se_square3 = np.ones((3, 3), np.uint8)
se_square5 = np.ones((5, 5), np.uint8)
se_square7 = np.ones((7, 7), np.uint8)
se_disk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
se_disk5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# Original image
original = image_bool.copy()

# ============ RESULT 1: NOISE REMOVAL ============
# Using opening (erosion followed by dilation)
result1_opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, se_square3)

# Alternative: Using area opening to remove small components
# Convert to uint8 for connected component labeling
image_uint8 = img_as_ubyte(image_bool)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_uint8)
result1_area_opening = np.zeros_like(image_uint8)
min_area = 50  # Minimum area to keep
for i in range(1, num_labels):  # Start from 1 to skip the background
    if stats[i, cv2.CC_STAT_AREA] > min_area:
        result1_area_opening[labels == i] = 255

# ============ RESULT 2: SHAPE SEPARATION ============
# Using erosion to separate connected components
result2_erosion = cv2.erode(image, se_square5, iterations=2)

# Alternative: Using watershed to separate touching objects
# First, distance transform
dist_transform = cv2.distanceTransform(image_uint8, cv2.DIST_L2, 5)
dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Threshold to get markers
_, markers = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY)
markers = cv2.connectedComponents(markers.astype(np.uint8))[1]

# Apply watershed
result2_watershed = image_uint8.copy()
# Expand image to 3 channels for watershed
result2_watershed_color = cv2.cvtColor(result2_watershed, cv2.COLOR_GRAY2BGR)
cv2.watershed(result2_watershed_color, markers.astype(np.int32))
result2_watershed = (markers > 0).astype(np.uint8) * 255

# ============ RESULT 3: FILLING HOLES ============
# Using closing (dilation followed by erosion)
result3_closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se_square5)

# Alternative: Using floodfill to fill holes
# Create a padded version to ensure border handling
padded = np.pad(image_uint8, 1, mode='constant')
mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
filled = padded.copy()
cv2.floodFill(filled, mask, (0, 0), 255, 0, 0, cv2.FLOODFILL_FIXED_RANGE)
filled = filled[1:-1, 1:-1]  # Remove padding
result3_floodfill = cv2.bitwise_not(filled) | image_uint8

# ============ RESULT 4: CONTOUR EXTRACTION ============
# Using morphological gradient (dilation - erosion)
result4_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, se_square3)

# ============ RESULT 5: SKELETONIZATION ============
# Create a skeleton using morphological operations
def skeletonize(img):
    skeleton = np.zeros_like(img)
    img_copy = img.copy()
    while cv2.countNonZero(img_copy) > 0:
        # Erosion
        eroded = cv2.erode(img_copy, se_square3)
        # Opening on eroded image
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, se_square3)
        # Difference between erosion and opening
        diff = cv2.subtract(eroded, opened)
        # Add the difference to the skeleton
        skeleton = cv2.bitwise_or(skeleton, diff)
        # Update the temporary image
        img_copy = eroded.copy()
    return skeleton

result5_skeleton = skeletonize(image)

# Alternative: Using scikit-image for skeletonization
img_bool = image_bool.astype(bool)
result5_skeleton_skimage = morphology.skeletonize(img_bool).astype(np.uint8) * 255

# Display results
display_images(
    [image, result1_opening, result1_area_opening, 
     result2_erosion, result2_watershed,
     result3_closing, result3_floodfill,
     result4_gradient, result5_skeleton],
    ['Original Image', 
     'Result 1a: Noise Removal (Opening)', 'Result 1b: Noise Removal (Area Opening)',
     'Result 2a: Shape Separation (Erosion)', 'Result 2b: Shape Separation (Watershed)',
     'Result 3a: Hole Filling (Closing)', 'Result 3b: Hole Filling (Floodfill)',
     'Result 4: Contour Extraction (Gradient)', 'Result 5: Skeletonization'],
    3, 3, figsize=(15, 15)
)

print("Explanation of morphological operations applied:")
print("\nResult 1: Noise Removal")
print("- Opening operation removes small isolated objects and noise")
print("- Area opening preserves objects larger than a specified threshold")

print("\nResult 2: Shape Separation")
print("- Erosion shrinks objects, separating connected components")
print("- Watershed segmentation separates touching objects based on distance transform")

print("\nResult 3: Hole Filling")
print("- Closing operation fills small holes and connects nearby objects")
print("- Floodfill technique fills holes by inverting background-filled image")

print("\nResult 4: Contour Extraction")
print("- Morphological gradient (dilation - erosion) extracts object boundaries")

print("\nResult 5: Skeletonization")
print("- Successive erosions while preserving topology creates a skeleton/medial axis")
