from google.colab import drive
drive.mount('/content/drive')

#Se installa la última versión de la librería Ultralytics
!pip install -U ultralytics

# Importación de librerías necesarias
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

#Exportación del modelo no pre-entrenado de Yolo11 en su versión l
model = YOLO('yolo11l.yaml')

#Entrenamiento del modelo
result=model.train(data="/content/drive/MyDrive/Proyecto_de_grado/data/data.yaml",
                   epochs=500,patience=450,batch=16,plots=True,optimizer="auto",lr0=1e-4,project="/content/drive/MyDrive/Proyecto_de_grado/runs/YOLO11/l")

#Técnica de validación al modelo entrenado. 
metrics = model.val(data='/content/drive/MyDrive/Proyecto_de_grado/data/data.yaml', project='/content/drive/MyDrive/Proyecto_de_grado/runs/YOLO11/l')
