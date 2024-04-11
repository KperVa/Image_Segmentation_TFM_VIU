from Detector import *
import cv2
import subprocess
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer() #Timer to count how much time spend the algorithm running. 
img_path = "images/img181.jpg"
vid_path = "images/New_York_inicio.mp4"

detector = Detector(model_type= "IS")
# resultado, img_size = detector.onImage(img_path)

resultado_vid = detector.onVideo(vid_path)
end = timeit.default_timer()
print(f"Tiempo de ejecución: {end - start} segundos")