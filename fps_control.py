import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg 
from PIL import Image
import cv2
import statistics

import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

frame_num = 0
fps_array = []
count=0
dtime = dict()
dwell_time=dict()
object_id_list = []
time_label=0
encoder = gdet.create_box_encoder('model_data/mars-small128.pb', batch_size=1)
tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None))
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
video_path ='omer-emre.mp4'
saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-tiny-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
vid = cv2.VideoCapture(video_path)



def party_value(frame):
    
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (416, 416))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.50
    )
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)
    pred_bbox = [bboxes, scores, classes, num_objects]
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    allowed_classes = list(class_names.values())
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)
    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
    #
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, 1.0, scores)
    detections = [detections[i] for i in indices] 
    return detections      
def fps_detect():
    global fps_array,x,vid,dtime,dwell_time,object_id_list,encoder,tracker,config,session,saved_model_loaded,infer
    for i in range(15):
        start_time = time.time()
        return_value, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        detections=party_value(frame)
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
        fps = 1.0 / (time.time() - start_time)
        fps_array.append(fps)
        x = statistics.mean(fps_array)
        


    return x
        
def yolo_detection(frame_value):
    global frame_num,x,vid,dtime,dwell_time,object_id_list,time_label,encoder,tracker,config,session,saved_model_loaded,infer
    while True:
        return_value, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        frame_num +=1
        print(frame_value)
        if frame_num%frame_value==0:
            detections=party_value(frame)     
            tracker.predict()
            tracker.update(detections)
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                
                if(int((bbox[0]+bbox[2])/2)>600):
                    if(int((bbox[1]+bbox[3])/2)>600):
                        id=track.track_id
                        if id not in object_id_list:
                            object_id_list.append(id)
                            dtime[id] = 1
                            dwell_time[id] = 1
                        else:
                            dtime[id] = +1
                            dwell_time[id] += dtime[id]
                        time_label = f'{int(dwell_time[id])}'
                        cv2.putText(frame, class_name + "-" + str(track.track_id)+"-"+str(int(float(float(time_label)/float(frame_value)))),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), (0,255,0), -1)
                        
                        if int(float(float(time_label)/float(frame_value)))>=5:
                            json= {
                                "cam_id":"1",
                                "frame_uuid":"165464546",
                                "danger":True,
                                "device_id":"Vamos Akademi",
                            }
                            print(json)
                    else:

                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), (0,255,0), -1)
                        cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

                else:

                    time_label=0
                    
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), (0,255,0), -1)
                    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            if True:
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                result=cv2.resize(result,(848,480))
                cv2.imshow("Output Video", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break



if __name__ == '__main__':
    system_fps=fps_detect()
    print(system_fps,"dönen fps")
    yolo_detection(int(system_fps))