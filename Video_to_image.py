import cv2 
import os

def extraer_imagenes(video_path, folder):
    video = cv2.VideoCapture(video_path)
    contador = 0

    while True:
        ret, frame = video.read()
        
        if not ret:
            break

        nombre_imagen = os.path.join(folder, f'test_frame_{contador}.png')
        cv2.imwrite(nombre_imagen, frame)
        contador += 1
    video.release()
    cv2.destroyAllWindows()

ruta_video = "New_York_largo.mp4"
ruta_folder = "New_York_test"

extraer_imagenes(ruta_video, ruta_folder)