from ultralytics import YOLO
import timeit
import cv2

start = timeit.default_timer() #Timer to count how much time spend the algorithm running. 
test_images = ["Dataset/test/1419283562_0574818000.png", "Dataset/test/test_frame_676.png"]
# Load the model
# model = YOLO("yolov8n-seg.pt")   
# model = YOLO("best_primero_n.pt") 
# model = YOLO("mejor_YOLOv8s.pt")
model = YOLO("mejor_YOLOv8n.pt")

# Run the model with an image or a video, comment on the unused one.
# model.predict(task="segment", show=True, conf=0.5, source='images/img181.jpg' )
# results = model.predict(task="segment", show=True, conf=0.5, source='images/New_York_peatones.mp4', stream= True )

# YOLO task=segment mode=predict model=best.pt show=True conf=0.25 source='images/New_York_inicio.mp4'

model.predict(task="segment", show=True, conf=0.25, source=test_images[1] )

end = timeit.default_timer()
print(f"Tiempo de ejecución: {end - start} segundos")




