from machine import Pin
import time
PUL = Pin(3, Pin.OUT)  # Pin para la señal de pulso
DIR = Pin(8, Pin.OUT)  # Define el pin de dirección
while True:  # Bucle infinito
    DIR.value(0)  # Configura la dirección del motor
    PUL.value(1)  # Activa el pulso
    time.sleep_us(450)  # Espera
    PUL.value(0)  # Desactiva el pulso
    time.sleep_us(450)  # Espera
