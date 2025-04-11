import machine
import utime
# Define los pines GPIO utilizados para controlar el driver TB6600
DIR_PIN = machine.Pin(6, machine.Pin.OUT)  # GPIO6 para dirección
STEP_PIN = machine.Pin(7, machine.Pin.OUT)  # GPIO7 para paso
# Configura los pines iniciales
DIR_PIN.off()   # Asegúrate de que el pin DIR esté en estado bajo para girar en una dirección
STEP_PIN.off()  # Asegúrate de que el pin STEP esté en estado bajo inicialmente
# Define la velocidad de paso (en Hz)
STEP_DELAY = 3  # 0.1 ms (10 microsegundos) entre pasos -> controla la velocidad
# Número de pasos para dar una vuelta completa (puede variar según tu motor)
STEPS_PER_REV = 200  # Ejemplo: motor NEMA 17 típico
# Función para hacer girar el motor en una dirección
def rotate_clockwise():
    DIR_PIN.on()  # Establece el sentido de giro en el sentido de las agujas del reloj
    for _ in range(int(STEPS_PER_REV)):
        STEP_PIN.on()
        utime.sleep_us(1000)
        STEP_PIN.off()
        utime.sleep_us(1000)
# Prueba de movimiento giratorio constante
while True:
    rotate_clockwise()
