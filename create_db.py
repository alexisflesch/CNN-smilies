"""
Open image files in the folder database, augment them, and save them as a PyTorch dataset.
"""

import os
import cv2
import numpy as np
import torch

# Define paths to dataset folders
happy_folder = "database/happy"
sad_folder = "database/sad"

# Define the output file name for the augmented dataset
output_file = "augmented_dataset.pth"

# Define augmentation parameters
rotation_angles = np.arange(-15, 16, 5)
# (x, y) translations in pixels
translations = [(i, j) for i in np.arange(-2, 2) for j in np.arange(-2, 2)]
stretch_factors = [(i, j) for i in np.arange(0.9, 1.1, 0.1)
                   for j in np.arange(0.9, 1.1, 0.1)]
# Initialize lists to store augmented images and labels
augmented_images = []
labels = []

# Function to perform image augmentation


def augment_image(image, label):
    # Resize the image to a consistent 32x32 shape
    image = cv2.resize(image, (32, 32))
    augmented_images.append(image)
    labels.append(label)
    # return

    # Rotate the image
    for angle in rotation_angles:
        rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D(
            (image.shape[1] / 2, image.shape[0] / 2), angle, 1.0), (32, 32))
        augmented_images.append(rotated_image)
        labels.append(label)

    # Translate the image
    for tx, ty in translations:
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, (32, 32))
        augmented_images.append(translated_image)
        labels.append(label)

    # Stretch the image
    for sx, sy in stretch_factors:
        stretched_image = cv2.resize(image, None, fx=sx, fy=sy)
        stretched_image = cv2.resize(stretched_image, (32, 32))
        augmented_images.append(stretched_image)
        labels.append(label)


# Iterate through the happy and sad folders
for label, folder in enumerate([happy_folder, sad_folder]):
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            # Read the image in grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            augment_image(image, label)

# Convert the list of images to a single numpy array
augmented_images = np.array(augmented_images, dtype=np.float32)

# Shuffle the dataset
shuffled_indices = np.arange(len(labels))
np.random.shuffle(shuffled_indices)
augmented_images = augmented_images[shuffled_indices]
labels = np.array(labels)[shuffled_indices]
labels = torch.tensor(labels, dtype=torch.int64)

# Create a PyTorch dataset
dataset = torch.utils.data.TensorDataset(
    torch.unsqueeze(torch.from_numpy(augmented_images), 1), labels)  # Add a single channel dimension

# Save the dataset as a PyTorch file
torch.save(dataset, output_file)

print(f"Augmented dataset saved to {output_file}")
print(f"Number of images: {len(augmented_images)}")
