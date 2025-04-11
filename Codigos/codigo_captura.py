import cv2
import datetime as dt



capture = cv2.VideoCapture(1,cv2.CAP_DSHOW)

capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
capture.set(cv2.CAP_PROP_AUTOFOCUS,0)
#capture.set(cv2.CAP_PROP_FPS,1000)
capture.set(cv2.CAP_PROP_FOCUS, 0)
capture.set(cv2.CAP_PROP_EXPOSURE, -7)
# Inicializar  tiempo
tiempoA = dt.datetime.now()
path = 'D:/Documentos/kevin/Programacion/Python/proyecto/fotos_trabajo/fot3'

while (capture.isOpened()):
    ret, frame = capture.read()
    if (ret == True):
        # Almacenar el tiempo actual
        tiempoB = dt.datetime.now()
        # Cuanto tiempo ha pasado desde tiempoA?
        tiempoTranscurrido = tiempoB - tiempoA
        cv2.imshow("webCam", frame)
        # Si han pasado 3 segundos ingresa al if
        if tiempoTranscurrido.seconds >= 2:
            cv2.imwrite(path + dt.datetime.now().strftime('IMG-%Y-%m-%d-%H%M%S') + '.jpg', frame)
            # Se debe encerar el tiempo trascurrido para voler a contar
            tiempoTranscurrido = 0
            # Se debe establecer un nuevo tiempoA
            tiempoA = dt.datetime.now()
        if (cv2.waitKey(1) == ord('s')):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
