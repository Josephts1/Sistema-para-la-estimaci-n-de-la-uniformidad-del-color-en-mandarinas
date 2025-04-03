  from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

#Código de segmentación
folder_path = '/content/drive/MyDrive/Proyecto_de_grado/runs/YOLO11/s/crops_para_histo_CIELAB/50_mandarinas_recortadas'
images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

images_segmentadas=[]
for i in range(len(images)):

  image = cv2.imread(images[i]); image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_rgb = image_rgb.astype(np.float64) / 255.0
  W_L = Foto.shape[0]/Foto.shape[1]
  if W_L<0.88:
    continue

  S = image_rgb[:, :, 0] / image_rgb[:, :, 2]
  mask = S > 0.999
  masked_Foto = image_rgb * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
  images_segmentadas.append(masked_Foto)

#Código para el cálculo del índice y clasificación de mandarinas a partir del índice.
mandarinas_data = []
#Acceso a imágenes segmentadas y su índice en la lista images_segmentadas
for idx,image_rgb in enumerate(images_segmentadas):
  #Conversión a espacio de color LAB
  image_lab = cv2.cvtColor(image_rgb.astype(np.float32), cv2.COLOR_RGB2Lab)

  #Conversión a escala de grises
  gray = cv2.cvtColor(image_rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
  #Aplicación de una máscara para eliminar del análisis los pixeles equivalentes a cero.
  mask = gray > 0

  #Separación de canales LAB
  L, A, B = cv2.split(image_lab)

  #Acceso a los valores diferentes de cero en la imagen
  L_mandarina = L[mask] #Va de 0 a 100
  A_mandarina = A[mask] #Va de -128 a 127
  B_mandarina = B[mask] #Va de -128 a 127

  #Cálculo del área de la elipse.
  ab_points = np.column_stack((A_mandarina, B_mandarina))
  ab_mean = np.mean(ab_points,axis=0)

  #Realce de canal a
  imp_a = np.abs(ab_points[:,0])

  #Matriz de covarianza y parámetros para la elipse
  cov = np.cov(ab_points, rowvar=False, aweights=imp_a)
  vals, vecs = np.linalg.eigh(cov)
  width, height = np.sqrt(vals)
  theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
  ellipse = Ellipse(xy=ab_mean, width=width, height=height, angle=theta, edgecolor='red', facecolor='none', lw=2)

  #Cálculo del índice
  Area_elipse = np.pi * vals[0] * vals[1]
  IC = 1/Area_elipse

  #Almacenamiento de datos
  mandarinas_data.append({
      'image_rgb':image_rgb,
      'image_lab':image_lab,
      'mask':mask,
      'A_mandarina':A_mandarina,
      'B_mandarina':B_mandarina,
      'std_a':std_a,
      'std_b':std_b,
      'ab_points':np.column_stack((A_mandarina, B_mandarina)),
      'Area_elipse':Area_elipse,
      'vals':vals,
      'ellipse':ellipse,
      'IC': IC,
      'index':idx
  })

#Ordenamiento de mandarinas de menor a mayor según IC
orden_mandarinas = sorted(mandarinas_data, key=lambda d: d['IC'], reverse=True)

fig, ax = plt.subplots(len(orden_mandarinas),3,figsize=(15,5*len(orden_mandarinas)),squeeze=False)

#Extracción de los datos ordenados
for i, data in enumerate(orden_mandarinas):
  image_rgb = data['image_rgb']
  image_lab = data['image_lab']
  mask = data['mask']
  A_mandarina = data['A_mandarina']
  B_mandarina = data['B_mandarina']
  ab_points = data['ab_points']
  std_a = data['std_a']
  std_b = data['std_b']
  Area_elipse = data['Area_elipse']
  vals = data['vals']
  ellipse = data['ellipse']
  IC = data['IC']
  original_index = data['index']

  ax[i,0].imshow(image_rgb)
  ax[i,0].set_title(f'Mandarina: {original_index} \n con eigenvalores e1: {vals[0]} \n e2: {vals[1]}')
  ax[i,0].axis('off')

  ax[i,1].hist(A_mandarina, bins=100, range=[-15,75], color="red", alpha=0.7,label="Canal a")
  ax[i,1].hist(B_mandarina, bins=100, range=[-15,75], color="blue", alpha=0.7,label="Canal b")
  ax[i,1].set_title(f'Histograma mandarina {original_index} \n std_a: {std_a} y std_b: {std_b}')
  ax[i,1].legend()

  ax[i,2].scatter(ab_points[:,0], ab_points[:,1], s=1,alpha=0.5)

  ax[i,2].add_patch(ellipse)
  ax[i,2].set_title(f'Puesto de clasificación: {i}\n area_ellipse: {Area_elipse}\n IC: {IC}')
  ax[i,2].set_xlabel('canal a')
  ax[i,2].set_ylabel('canal b')

  ax[i,2].set_xlim([-15,75])
  ax[i,2].set_ylim([-15,75])

fig.tight_layout(h_pad=4,w_pad=2)
plt.show()
