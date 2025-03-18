#Uso de google drive para el almacenamiento de datos.
from google.colab import drive
drive.mount('/content/drive')

#Importación de librerías a utilizar
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Carpeta con mandarinas recortadas por modelo YOLO
folder_path = '/content/drive/MyDrive/Proyecto_de_grado/runs/YOLO11/s/crops_para_histo_CIELAB/50_mandarinas_recortadas'

#Almacenamiento y detección de rutas para cada imagen
images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
print(f'# imágenes en carpeta: {len(images)}')


images_segmentadas=[]
for i in range(len(images)):
  #Lectura de todas las imágenes en la carpeta
  image = cv2.imread(images[i]);#print(image[i].shape)
  #Conversión de BGR a RGB
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB);#print(image_rgb[i].shape)
  image_rgb = image_rgb.astype(np.float64) / 255.0
  #Cálculo de relación entre las dimensiones de cada imagen. (Altura/Ancho)
  W_L = image_rgb.shape[0]/image_rgb.shape[1]

  #Eliminación de mandarinas que no se visualizan completamente. Elimina las imágenes con relación altura/ancho menor a la óptima
  if W_L<0.88:
    continue

  fig, axes = plt.subplots(1, 4, figsize=(15,5))
  
  #Visualización imagen original
  axes[0].imshow(image_rgb)
  axes[0].set_title(f'Imagen Original:{i}')
  axes[0].axis('on')

  #Calculo de la relación entre la primera capa (R) y la tercera capa (B) para la segmentación
  S = image_rgb[:, :, 0] / image_rgb[:, :, 2]
  
  #Creación de máscara binaria para pixeles mayores al threshold. Almacena posiciones
  mask = S > 0.99
  axes[1].imshow(mask,cmap='gray')
  axes[1].set_title('Mask')
  axes[1].axis('on')

  #Aplicación de la máscara en los tres canales
  masked_Foto = image_rgb * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
  
  images_segmentadas.append(masked_Foto); #print (images_segmentadas.shape)
  
  #Visualización de imagen con máscara aplicada.
  axes[2].imshow(masked_Foto)
  axes[2].set_title(f'Imagen segm {i}')
  axes[2].axis('on')
  
  #Graficación del histograma de la división R/B
  axes[3].hist(S.ravel(), bins=50, color='red', alpha=0.75, range=(0,8))
  axes[3].set_title('Histograma de R/B')
  axes[3].set_xlabel('R/B')
  axes[3].set_ylabel('Frecuencia')
  plt.show()
