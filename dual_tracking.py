# MIT License
# Copyright (c) 2019 Brainjar
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV and a Google Coral USB accelerator
# Drivers for the camera and OpenCV are included in the base image of the Jetson Nano

import cv2
import os
import numpy as np
import time
import threading
import random
from ash import Ash
from edgetpu.detection.engine import DetectionEngine
from tftrt_helper import FrozenGraph, TfEngine, TftrtEngine
from keras.models import load_model
from camera_helper import gstreamer_pipeline

execution_path = os.getcwd()

quit = False

width = 1280
height = 720
channels = 3
view_width = 1280
view_height = 720

WINDOW_TITLE = 'PiCarChu'
FRAME_RATE = 30

scaler = np.array([width, height, width, height])

model_path = os.path.abspath(os.path.join(execution_path, 'model/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'))
label_path = os.path.abspath(os.path.join(execution_path, 'model/coco_labels.txt'))

engine = DetectionEngine(model_path)


def bboxscale(box):  # scales a bounding box
    ret = box * scaler
    return ret

def shower():
    global quit
    global a
    global pfps
    global latestFrame

    time.sleep(2)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    delay = 1/FRAME_RATE
    while not quit:
        output_frame = np.copy(latestFrame)
        if a.ashed == 1:
            cv2.putText(output_frame, a.name, a.Get_centroid_tuple(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, a.color, 1)
            cv2.putText(output_frame, str(a.reid_score), a.Get_p1_tuple(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, a.color, 1)
            cv2.rectangle(output_frame, a.Get_p1_tuple(), a.Get_p2_tuple(), a.color, 1)

        cv2.putText(output_frame, "fps:{:0>3d}".format(pfps), (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1)
        cv2.imshow(WINDOW_TITLE, output_frame)
        time.sleep(delay)

    cv2.destroyAllWindows()

def getter():
    global quit
    global latestFrame
    global lastrun
    cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=width,
                                              capture_height=height,
                                              display_width=width,
                                              display_height=height,
                                              framerate=FRAME_RATE,
                                              flip_method=2), cv2.CAP_GSTREAMER)
    #delay = 1 / FRAME_RATE
    if cap.isOpened():
        while not quit:
            _,latestFrame = cap.read()
            if(time.time() - lastrun > 20):    # just so that the camera gets released on crash
                print('camera closed')
                cap.release()
                break
            #time.sleep(delay)

        cap.release()

def process():
    global quit
    global lastrun
    global latestFrame
    global a
    global pfps

    time.sleep(2)

    s = time.time() # keep start time of first frame
    while not quit:
        '''
        Speed loss is definately in the first part, the rest is all sub ms
        The detection is the only thing usualy taking more then 10-ish ms
        '''
        tensorframe = cv2.resize(latestFrame, (300, 300))  # resize to model input size - linear
        tensorframe = cv2.cvtColor(tensorframe, cv2.COLOR_BGR2RGB)  # swap channels
        tensorframe = np.reshape(tensorframe, (300 * 300 * 3))  # flatten

        results = engine.DetectWithInputTensor(tensorframe, threshold=0.6, top_k=15)  # run inference

        personrects = []
        for result in results:
            if result.label_id == 0 :
                box = bboxscale(result.bounding_box.flatten())      # Flatten the boxes of persons
                personrects.append(box)                             # add to list

        lastrun = time.time()     # get endtime of frame
        t = lastrun - s           # calculate duration of last frame
        if t > 0.4:
            print('bekke traag')
        s = lastrun               # start of next frame is end of last
        a.Update(personrects, t*1000, latestFrame)    # update the Ash object

        pfps = int(0.4* (1.0 / t) + 0.6*pfps)     # current fps

        keyCode = cv2.waitKey(1) & 0xff
        # Stop the program on the ESC key,
        # start tracking on space
        if keyCode == 27:
            quit = True
        elif keyCode == 32:
            if len(personrects) > 0:
                a.Asher(personrects, 1280, 720) # dirty af, but defines the Ash
                a.Set_Ref_Mbed(latestFrame)
            else:
                print('nobody in view, try again later.')


if __name__ == '__main__':
    global latestFrame
    global lastrun
    global pfps
    global a

    latestFrame = np.zeros((height, width, channels), dtype='uint8')

    turbonet = load_model('model/TurboNet_Siamese-retrain.hdf5')
    turbonet.summary()

    a = Ash(model=turbonet, name='Turbo')

    pfps = 0

    lastrun = time.time()

    window_title = 'PiCarChu'

    getter_t = threading.Thread(target=getter)
    getter_t.daemon = True
    getter_t.start()

    shower_t = threading.Thread(target=shower)
    shower_t.daemon = False
    shower_t.start()

    process()