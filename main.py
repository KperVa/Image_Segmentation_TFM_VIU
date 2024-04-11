import json
import os
import matplotlib.pyplot as plt
import cv2
import numpy

from Detector import *

def project_point(point, projection_matrix):
    """Projects 3D point into image coordinates

    Parameters
    ----------
    p1: iterable
        (x, y, z), line start in 3D
    p2: iterable
        (x, y, z), line end in 3D
    width: float
           width of marker in cm
    projection matrix: numpy.array, shape=(3, 3)
                       projection 3D location into image space

    Returns (x, y)
    """
    point = numpy.asarray(point)
    projection_matrix = numpy.asarray(projection_matrix)

    point_projected = projection_matrix.dot(point)
    point_projected /= point_projected[2]

    return point_projected

def _filter_lanes_by_size(label, min_height=40):
    """ May need some tuning """
    filtered_lanes = []
    for lane in label['lanes']:
        lane_start = min([int(marker['pixel_start']['y']) for marker in lane['markers']])
        lane_end = max([int(marker['pixel_start']['y']) for marker in lane['markers']])
        if (lane_end - lane_start) < min_height:
            continue
        filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes

def _filter_few_markers(label, min_markers=2):
    """Filter lines that consist of only few markers"""
    filtered_lanes = []
    for lane in label['lanes']:
        if len(lane['markers']) >= min_markers:
            filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes


def _fix_lane_names(label):
    """ Given keys ['l3', 'l2', 'l0', 'r0', 'r2'] returns ['l2', 'l1', 'l0', 'r0', 'r1']"""

    # Create mapping
    l_counter = 0
    r_counter = 0
    mapping = {}
    lane_ids = [lane['lane_id'] for lane in label['lanes']]
    for key in sorted(lane_ids):
        if key[0] == 'l':
            mapping[key] = 'l' + str(l_counter)
            l_counter += 1
        if key[0] == 'r':
            mapping[key] = 'r' + str(r_counter)
            r_counter += 1
    for lane in label['lanes']:
        lane['lane_id'] = mapping[lane['lane_id']]


def read_json(json_path, min_lane_height=20):
    """ Reads and cleans label file information by path"""
    with open(json_path, 'r') as jf:
        label_content = json.load(jf)

    _filter_lanes_by_size(label_content, min_height=min_lane_height)
    _filter_few_markers(label_content, min_markers=2)
    _fix_lane_names(label_content)

    content = {
        'projection_matrix': label_content['projection_matrix'],
        'lanes': label_content['lanes']
    }

    for lane in content['lanes']:
        for marker in lane['markers']:
            for pixel_key in marker['pixel_start'].keys():
                marker['pixel_start'][pixel_key] = int(marker['pixel_start'][pixel_key])
            for pixel_key in marker['pixel_end'].keys():
                marker['pixel_end'][pixel_key] = int(marker['pixel_end'][pixel_key])
            for pixel_key in marker['world_start'].keys():
                marker['world_start'][pixel_key] = float(marker['world_start'][pixel_key])
            for pixel_key in marker['world_end'].keys():
                marker['world_end'][pixel_key] = float(marker['world_end'][pixel_key])
    return content



# images_path = "Dataset/Images"
images_path = "Dataset/val"
# images_path = "Dataset/test"
labels_path = "Dataset/val_json/"
images_list = os.listdir(images_path)
numpy.set_printoptions(linewidth=numpy.inf)

# ### En la primera ejecución, esta parte renombra las imágenes. 
# for x in images_list:
#     original_name = images_path +'/' + x
#     new_name = images_path +'/' + x[:-15] + '.png'
#     os.rename(original_name, new_name)
#     print(f"Archivo {x} renombrado")
# ###

for n, image in enumerate(images_list):
    img = cv2.imread(images_path + "/" + image)
    img_shape = img.shape
    data_path = labels_path + image[:-4] + '.json'
    data = read_json(data_path)
    width = 0.1
    output = []
    p_matrix = numpy.asarray(data["projection_matrix"])
    for l, lane in enumerate(data["lanes"]):
        for i, marker in enumerate(lane["markers"]):
            x1,y1,z1 = marker["world_start"]["x"], marker["world_start"]["y"], marker["world_start"]["z"]
            p1 = numpy.array([x1, y1, z1])
            x2,y2,z2 = marker["world_end"]["x"], marker["world_end"]["y"], marker["world_end"]["z"]
            p2 = numpy.array([x2, y2, z2])
            p1_projected = project_point(p1, p_matrix)
            p2_projected = project_point(p2, p_matrix)
            points = numpy.zeros((4, 2), dtype=numpy.float32)
            shift_multiplier = 1  # simplified

            projected_half_width1 = p_matrix[0, 0] * width / p1[2] / 2.0
            points[0, 0] = ((p1_projected[0] - projected_half_width1) * shift_multiplier)/img_shape[1]
            points[0, 1] = (p1_projected[1] * shift_multiplier)/img_shape[0]
            points[1, 0] = ((p1_projected[0] + projected_half_width1) * shift_multiplier)/img_shape[1]
            points[1, 1] = (p1_projected[1] * shift_multiplier)/img_shape[0]

            projected_half_width2 = p_matrix[0, 0] * width / p2[2] / 2.0
            points[2, 0] = ((p2_projected[0] + projected_half_width2) * shift_multiplier)/img_shape[1]
            points[2, 1] = (p2_projected[1] * shift_multiplier)/img_shape[0]
            points[3, 0] = ((p2_projected[0] - projected_half_width2) * shift_multiplier)/img_shape[1]
            points[3, 1] = (p2_projected[1] * shift_multiplier)/img_shape[0]
        
            points = numpy.reshape(points, (1,8))
            l_points = points.tolist()
            l_points = [80] + l_points
            l_points = str(l_points)
            l_points = l_points.replace(",","")
            l_points = l_points.replace("[","")
            l_points = l_points.replace("]","")
            output.append(l_points)
            
    #### Esta parte es la que detecta el resto de coches de la imagen
    detector = Detector(model_type= "IS")
    res, image_size = detector.onImage(images_path + "/" + image)
    for i,k in zip(res["instances"].pred_classes, res["instances"].pred_masks):
        cpu_l = k.cpu().numpy()
        cpu_c = i.cpu().numpy()
        mask = (cpu_l * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_l = []
        for points in contours[0]:
            x,y = points[0]
            x = x/img_shape[1]
            y = y/img_shape[0]
            contours_l.append(x)
            contours_l.append(y)
        lista_c = [cpu_c.tolist()]
        lista = lista_c + contours_l
        lista = str(lista)
        lista = lista.replace(",","")
        lista = lista.replace("[","")
        lista = lista.replace("]","")
        output.append(lista)

        # Crear y abrir el archivo en modo escritura ('w')
    with open(labels_path + image[:-4] +".txt" , 'w') as f:
        for fila in output:
            fila_str = str(fila)
            fila_str = fila_str.replace(",","")
            fila_str = fila_str.replace("'","")
            fila_str = fila_str.replace("[","")
            fila_str = fila_str.replace("]","")
            label = fila_str + "\n"
            f.write(label)
    if n%1000 == 0:
        print(f"Json de la imagen {n} generado:" )