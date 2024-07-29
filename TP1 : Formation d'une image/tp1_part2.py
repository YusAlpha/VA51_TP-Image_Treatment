import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

gray_scale_img_path = "grayscale_img.png"
image = cv2.imread(gray_scale_img_path)

# Liste des niveaux de gris souhaités
niveaux_de_gris = [128, 64, 32, 16, 8, 4, 2]

fig, axs = plt.subplots(2, len(niveaux_de_gris)+1, figsize=(15, 5))



# Afficher l'image d'origine à gauche
axs[0][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0][0].set_title("Image d'origine")


# Boucle pour créer et enregistrer des images pour chaque niveau de gris
for col,  niveau in enumerate(niveaux_de_gris):
    col += 1
    
    # Créer une copie de l'image en niveaux de gris
    image_niveau_gris = image.copy()
    
    # Appliquer la quantification à ce niveau de gris
    seuil = 256 // niveau  # Calculer le seuil pour la quantification
    image_niveau_gris = (image_niveau_gris // seuil) * seuil
    
    # Enregistrer l'image avec le niveau de gris spécifié
    cv2.imwrite(f'transformed_images/image_{niveau}_niveaux_gris.jpg', image_niveau_gris)
    
    # Afficher l'image convertie
    axs[0, col].imshow(image_niveau_gris, cmap='gray')
    axs[0, col].set_title(f"Niveau de gris {niveau}")
    
    hist = cv2.calcHist([image_niveau_gris], [0], None, [256], [0, 256])
    axs[1, col].imshow(hist, cmap='gray')
    axs[1, col].set_title(f"H{niveau}")
    


# # Masquer les axes
# for ax in axs:
#     ax.axis('off')

# Afficher la figure
plt.tight_layout()
plt.show()

# cv2.imshow("Original Image", image)
# cv2.waitKey(0)
