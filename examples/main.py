# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

import argparse
from pathlib import Path
import sys
import subprocess
import threading
import logging

import asyncio
import websockets
import cv2
import base64
import torch
import numpy
import numpy as np
from timeit import default_timer as timer
import time
import random
import math
from datetime import datetime
import json

import calibrations_functions

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import TestRequirements
from boxmot.utils.torch_utils import select_device

__tr = TestRequirements()
__tr.check_packages(('ultralytics==8.0.124',))  # install

from detectors.strategy import get_yolo_inferer
from detectors.yolo_processor import Yolo
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.engine.model import TASK_MAP, YOLO
from ultralytics.yolo.utils import IterableSimpleNamespace, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.plotting import save_one_box
from utils import write_MOT_results

from boxmot.utils import EXAMPLES

# detected object indexes handling
detectedObjectIndexes = []

# Websocket communication
# Flag to signal the thread to stop
iterateee = 0
globalFrame = None

detectedObjectIdInCounter = []
countCar = 0
countBus = 0
countTruck = 0
countTrainTram = 0

killAsyncThreadForChar = False
stop_websocket_thread = False

#detectedObjects = [1]
homographyMatrix = []
perspectiveView = None
topDownView = None
cameraMatrix = None
distCoeffs = None
gps_coords = []
pixel_coords = []

ObjectListJson = {
          "AnalyticsId": 0,
          "CubeId": 0,
          "EvaluationTimestamp": str(int(time.time())*1000),
          "Failure": False,
          "FailureState": "None",
          "Id":"Brno",
          "MessageCounter": 0,
          "Objects": [],
          "supportedClassesForCounter": ["Car", "Bus", "Truck", "Train/Tram"],
          "totalObjectsCountForCounter": [],
          "Part": 1,
          "SinkId": 0,
          "TotalParts": 1
        }

# for StatisticsData
async def sendStatisticsData(uri):
    tryToConnectCount = 120
    while tryToConnectCount > 0:  
        try:
            async with websockets.connect(uri) as websocket:            
                while not stop_websocket_thread:        

                    ObjectListJson["totalObjectsCountForCounter"] = [countCar, countBus, countTruck, countTrainTram]
                    print(f'{ObjectListJson["Objects"]}')

                    data = {"ObjectList": ObjectListJson}
                    json_data = json.dumps(data)
                    await websocket.send(json_data)
                    #print(f"Sent data to server: {json_data}")
                    print(f"Sending data to server")
                    tryToConnectCount = 120
                    await asyncio.sleep(1)

        except Exception as e:
            print(f"Connection failed: {e}, retrying in 5 seconds...")
            tryToConnectCount = tryToConnectCount - 1
            await asyncio.sleep(30)

# for StatisticsData
def runSendingStatistics_thread():
    # Yunex websocket url
    server_uri = "ws://192.168.0.121:8000"

    # Local testing websocket url
    #server_uri = "ws://127.0.0.1:8000"
    asyncio.new_event_loop().run_until_complete(sendStatisticsData(server_uri))

################################################################
################################################################
################################################################

# for HorizontalChart 
async def sendDataToHorizontalChart(websocket, path):
    #global stop_websocket_thread
    while not stop_websocket_thread:
        #data = {
        #    "supportedClassesForCounter": ["Car", "Bus", "Truck", "Train/Trammm"],
        #    "totalObjectsCountForCounter": [countCar, countBus, countTruck, countTrainTram]
        #}
        ObjectListJson["totalObjectsCountForCounter"] = [countCar, countBus, countTruck, countTrainTram]

        now = datetime.now()
        #current_time = now.strftime("async thread: %H:%M:%S")
        current_time = now.strftime("%H:%M:%S")
        #print(ObjectListJson)
        print("[{}] Sending a socket/json".format(current_time))
        #print(stop_websocket_thread)

        await websocket.send(json.dumps(ObjectListJson))
        await asyncio.sleep(1)  # Send updates every second
    exit()

# for HorizontalChart
def runHorizontalChart_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(sendDataToHorizontalChart, "127.0.0.1", 8051)
    loop.run_until_complete(start_server)
    loop.run_forever()

################################################################
################################################################
################################################################

# for VideoStream
async def sendDataToVideoStreamViewer(websocket, path):
    #global stop_websocket_thread
    while not stop_websocket_thread:
        global globalFrame
        frame = globalFrame

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Process the frame as needed here
        _, image_data = cv2.imencode('.jpg', frame)
        image_data_base64 = base64.b64encode(image_data.tobytes()).decode('utf-8')
        
        now = datetime.now()
        #current_time = now.strftime("async thread: %H:%M:%S")
        current_time = now.strftime("%H:%M:%S")
        print("[{}] Sending a socket/json".format(current_time))
        #print(stop_websocket_thread)

        await websocket.send(image_data_base64)
        # Wait for 5 seconds before sending the next data
        await asyncio.sleep(1) 
    exit()

# for VideoStream
def runVideoStream_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(sendDataToVideoStreamViewer, "127.0.0.1", 8050)
    loop.run_until_complete(start_server)
    loop.run_forever()

class CameraData:
    def __init__(self):
        self.img = None
        self.camera_idx = 0
        self.window_name = ""
        self.win_pos = None
        self.H = None # Homograficka matica pouzivana v pocitacovom videni na vykonavanie transformacii obrazu, napriklad pri kalibracii kamery alebo pri mapovani obrazu na ine plochy.
        self.K = None # Vnutorna kalibracna matica kamery. Pouziva sa na definovanie vnutornych parametrov kamery, ako su ohniskova vzdialenost a opticke stredisko.
        self.D = None # Parameter pre distorcne koeficienty kamery, ktore sa pouzivaju na opravu deformacie obrazu sposobenej charakteristikami sosovky kamery.
        self.P = None # Projekcna matica, ktora sa pouziva v pocitacovom videni na transformaciu 3D bodov sveta do 2D obrazoveho priestoru.

class cameraParams:
    def __init__(self, H=None, K=None, D=None, P=None):
        self.H = H if H is not None else cv2.Mat()
        self.K = K if K is not None else cv2.Mat()
        self.D = D if D is not None else cv2.Mat()
        self.P = P if P is not None else cv2.Mat()

# global camera reference data
cam_ref_win_name = "cam_ref"
cd_ref = CameraData()
img_ref = None

# # # TESTING KIAC
def computePointInReferenceFrame(x, y, index):
    #global perspectiveView, topDownView, cameraMatrix, distCoeffs
    global cameraMatrix, distCoeffs

    newPoint = (x, y)

    # Draw distorted point
    transformedPoint = transformPointFromPerspectiveToTopDownView(newPoint, index)
    #cv2.circle(topDownView, (int(transformedPoint[0]), int(transformedPoint[1])), 2, (0, 0, 255), -1)
    #cv2.circle(topDownView, (int(transformedPoint[0]), int(transformedPoint[1])), 12, (0, 0, 255), 2)

    # Draw undistorted point
    distortedPoints = np.array([newPoint], dtype=np.float32)

    # Compensate distortion for the given point
    undistortedPoints = cv2.undistortPoints(distortedPoints, cameraMatrix, distCoeffs, P=cameraMatrix)

    # The output point is the first point in the undistorted array
    undistortedPoint = tuple(undistortedPoints[0, 0])

    transformedPointUndistorted = transformPointFromPerspectiveToTopDownView(undistortedPoint, index)
    #cv2.circle(perspectiveView, (int(newPoint[0]), int(newPoint[1])), 2, (255, 0, 0), -1)
    #cv2.circle(perspectiveView, (int(newPoint[0]), int(newPoint[1])), 12, (255, 0, 0), 2)
    #cv2.circle(topDownView, (int(transformedPoint[0]), int(transformedPoint[1])), 2, (0, 255, 0), -1)
    #cv2.circle(topDownView, (int(transformedPoint[0]), int(transformedPoint[1])), 12, (0, 255, 0), 2)
    return (int(transformedPoint[0]), int(transformedPoint[1]))

def transformPointFromPerspectiveToTopDownView(point, index):
    #print("Index: ", index)
    global homographyMatrix
    # Convert point to numpy array with shape (1, 1, 2) for cv2.perspectiveTransform
    inputPoints = np.array([[point]], dtype=np.float32)
    
    # Apply perspective transformation
    #outputPoints = cv2.perspectiveTransform(inputPoints, homographyMatrix)
    outputPoints = cv2.perspectiveTransform(inputPoints, homographyMatrix[index])
 
    # Convert output to desired format (x, y) tuple
    transformedPoint = tuple(outputPoints[0, 0])
    
    return transformedPoint
# # # TESTING KIAC

# To measure a distance in meters between two GPS coordinates, 
# you can use the Haversine formula. 
# The Haversine formula calculates the great-circle distance between 
# two points on the Earth's surface, given their 
# latitude and longitude in degrees. 
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Radius of the Earth in meters (mean value)
    radius = 6371000  # Approximately 6,371 km
    
    # Calculate the distance
    distance = radius * c
    return distance

# Define the four pairs of corresponding points
gps_coords = np.array([
    (49.22151, 16.58512),
    (49.22158, 16.58515),
    (49.22162, 16.58531),
    (49.22159, 16.58535),
    (49.22149, 16.58549),
    (49.22140, 16.58545),
    (49.22136, 16.58530)
])

pixel_coords = np.array([
    (708, 395),
    (726, 374),
    (1151, 374),
    (1172, 395),
    (1250, 683),
    (1131, 795),
    (726, 795)
])

pixelToGpsHomography = None

# Define a function to convert pixel coordinates to GPS coordinates
def pixel_to_gps(x, y):
    #print(x, y)
    pixel_point = np.array([[x, y, 1]])
    global pixelToGpsHomography
    #print(pixelToGpsHomography)
    gps_point = np.dot(pixelToGpsHomography, pixel_point.T)
    gps_point /= gps_point[2]
    return gps_point[0, 0], gps_point[1, 0]

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cd = param

        point = (x, y)
        camera_id = cd.camera_idx

        # draw actual click point to image
        cv2.circle(cd.img, point, 2, (0, 0, 255), 2)

        # compute point in reference image 
        point_to_ref = calibrations_functions.getPointInRefPlane(point, cd.H)
        #print(cd.H)

        # draw actual click point projected to reference image
        cv2.circle(cd_ref.img, point_to_ref, 2, (0, 0, 255), 2)

        # compute real world point coordinates and print it to terminal
        xyz_vec = calibrations_functions.getPositionXYZ(cd.P, cd_ref.P, point, point_to_ref)

        for xyz in xyz_vec:
            print(f"Point in real world: {xyz} m.")

def computePointInReferenceFrameOLD(x, y, param):
    cd = param

    point = (x, y)
    camera_id = cd.camera_idx

    # compute point in reference image 
    point_to_ref = calibrations_functions.getPointInRefPlane(point, cd.H)
    #print(cd.H)

    return point_to_ref

class Rect():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains(self, point_x, point_y):
        if self.x <= point_x <= self.x + self.width and self.y <= point_y <= self.y + self.height:
            return True
        else:
            return False

class DetectedObject:
    def __init__(self, id, className, x, y, width, height, detectedTime, gpsPosition):
        self.id = id
        self.className = className
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.ttl = 20
        self.objectTrajectory = []
        self.objectTrajectory.append((x, y))
        self.centerX = x + (width/2)
        self.centerY = y + (height/2)
        self.detectedTime = detectedTime
        self.realWorldPosition = 0
        self.gpsPosition = gpsPosition
        self.speed = 0

        self.red = random.randrange(0, 255)
        self.green = random.randrange(0, 255)
        self.blue = random.randrange(0, 255)

    def draw(self, image):
            cv2.rectangle(image, (self.x, self.y), (self.x + self.width, self.y + self.height), (0, 255, 0), 2)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def setRect(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def getRect(self):
        return (self.x, self.y, self.width, self.height)

    def getRectCenter(self):
        return (self.centerX, self.centerY)

    def getClassName(self):
        return self.className

    def getId(self):
        return self.id

    def getDetectedTime(self):
        return self.detectedTime

    def getRealWorldPosition(self):
        return self.realWorldPosition

    def getGpsPosition(self):
        return self.gpsPosition

    def setSpeed(self, speed):
        self.speed = speed

    def getSpeed(self):
        #self.speed = random.uniform(25.1, 29.9)
        #return (self.speed/10)/4
        #return 0
        #r = random.randint(27, 31)
        #if r == 31:
        #    return -1
        #else:
        #    return r
        return self.speed

    def decreaseTTL(self):
        self.ttl = self.ttl - 1

    def resetTTL(self):
        self.ttl = 20

    def getTTL(self):
        return self.ttl

    def updateObject(self, className, x, y, detectedTime, gpsPosition):
        self.className = className
        self.x = x
        self.y = y
        self.centerX = x + (self.width/2)
        self.centerY = y + (self.height/2)
        self.objectTrajectory.append((self.centerX, self.centerY))
        self.realWorldPosition = 0
        #self.detectedTime = detectedTime
        #self.gpsPosition = gpsPosition

        # Duration time between last two detections
        timeMeasurementDiration = detectedTime - self.detectedTime
        # Update detection time
        self.detectedTime = detectedTime

        # Distance between last two detections
        #gpsPosition[0] ---> latitude
        #gpsPosition[1] ---> longitude
        distance = haversine(self.gpsPosition[0], self.gpsPosition[1], gpsPosition[0], gpsPosition[1])
        # Update detection gps position
        self.gpsPosition = gpsPosition

        #speed = (distance/timeMeasurementDiration)*3.6
        speed = (distance/timeMeasurementDiration)
        self.setSpeed(speed)

    def getObjectTrajectory(self):
        return self.objectTrajectory


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    predictor.args.tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (opt.tracking_method + '.yaml')
    for iii in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.device,
            predictor.args.half,
            predictor.args.per_class
        )
        predictor.trackers.append(tracker)

def printHelp():
    print("aaaaaa")
    #print("python main.py --device 0 --source F:\Yunex_dataset_2023\xMiladyHorakove\test.mp4 --yolo-model yolov8m.pt --img 640 --tracking-method bytetrack --reid-model mobilenetv2_x1_0_market1501.pt --half")

@torch.no_grad()
def runProcessing(args):

    cam_params_file_name = "cameraParams/crossroadX_cameras_params.xml"
    #testing new params matrix
    #cam_params_file_name = "cameraParams/kolisteMHorakove_cameras_params.xml"

    cd_1 = CameraData()
    cd_2 = CameraData()
    cd_3 = CameraData()
    cd_4 = CameraData()
    cd_vec = []

    # load imgs, idxs etc...
    cd_ref.camera_idx = 0
    #topDownImageFrame = "cameraViews/kolisteMHorakove_topDownView.png"
    #topDownImageFrame = "cameraViews/crossroadX_birdView.png"
    topDownImageFrame = "cameraViews/kolisteMHorakove_topDownView_zoom.PNG"
    cd_ref.img = cv2.imread(topDownImageFrame)
    cd_ref.window_name = cam_ref_win_name
    cd_ref.win_pos = (500, 500)
    cd_vec.append(cd_ref)

    cd_1.camera_idx = 1
    #cd_1.img = cv2.imread("cameraViews/kolisteMHorakove_cam_ip32.png")
    cd_1.img = cv2.imread("cameraViews/crossroadX_cam_01.png")
    cd_1.window_name = "cam_1"
    cd_1.win_pos = (0, 0)
    cd_vec.append(cd_1)

    cd_2.camera_idx = 2
    #cd_2.img = cv2.imread("cameraViews/kolisteMHorakove_cam_ip34.png")
    cd_2.img = cv2.imread("cameraViews/crossroadX_cam_02.png")
    cd_2.window_name = "cam_2"
    cd_2.win_pos = (0, 0)
    cd_vec.append(cd_2)

    # PERSPECTIVE TO PLANE
    H_vec, K_vec, D_vec, P_vec = [], [], [], []

    # read params from files
    fs = cv2.FileStorage(cam_params_file_name, cv2.FILE_STORAGE_READ)
    H_node = fs.getNode("H_vec")
    K_node = fs.getNode("K_vec")
    D_node = fs.getNode("D_vec")
    P_node = fs.getNode("P_vec")

    for vectorIndex in range(H_node.size()):
        H_vec.append(H_node.at(vectorIndex).mat())
        K_vec.append(K_node.at(vectorIndex).mat())
        D_vec.append(D_node.at(vectorIndex).mat())
        P_vec.append(P_node.at(vectorIndex).mat())

    fs.release()

    # initialization through all cameras
    for j in range(len(cd_vec)):
        cd_vec[j].H = H_vec[j]
        cd_vec[j].K = K_vec[j]
        cd_vec[j].D = D_vec[j]
        cd_vec[j].P = P_vec[j]

    # Calculate the homography matrix
    global pixel_coords
    global gps_coords 
    global pixelToGpsHomography
    pixelToGpsHomography, _ = cv2.findHomography(np.array(pixel_coords), np.array(gps_coords))

    #birdViewPath = "D:\\DNN\\\yunex_yolo_object_tracking\\examples\\cameraViews\\crossroadX_birdView.png"
    birdViewPath = topDownImageFrame
    birdViewFrame = cv2.imread(birdViewPath, cv2.IMREAD_COLOR)

    model = YOLO(args['yolo_model'] if 'v8' in str(args['yolo_model']) else 'yolov8n')
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)

    # init detectedObjectIndexes array
    #initDetectedObjectIndexes()
    detectedObjectIndexesCounter = 0

    # detected objects
    #allowedDetectedObjects = ["person", "car", "truck", "bus", "tram/train", "bicycle", "motorcycle"]
    allowedDetectedObjects = ["car"]
    detectedObjects = []

    # extract task predictor
    predictor = model.predictor

    # combine default predictor args with custom, preferring custom
    combined_args = {**predictor.args.__dict__, **args}
    # overwrite default args
    predictor.args = IterableSimpleNamespace(**combined_args)
    predictor.args.device = select_device(args['device'])
    LOGGER.info(args)

    # setup source and model
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)
    predictor.setup_source(predictor.args.source)

    predictor.args.imgsz = check_imgsz(predictor.args.imgsz, stride=model.model.stride, min_dim=2)  # check image size
    predictor.save_dir = increment_path(Path(predictor.args.project) /
                                        predictor.args.name, exist_ok=predictor.args.exist_ok)

    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (predictor.save_dir / 'labels' if predictor.args.save_txt
         else predictor.save_dir).mkdir(parents=True, exist_ok=True)

    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(
            imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz)
        )
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = (
        0,
        [],
        None,
        (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
    )
    predictor.add_callback('on_predict_start', on_predict_start)
    predictor.run_callbacks('on_predict_start')

    yolo_strategy = get_yolo_inferer(args['yolo_model'])
    #yolo_strategy = yolo_strategy(
    #    model=model.predictor.model if 'v8' in str(args['yolo_model']) else args['yolo_model'],
    #    device=predictor.device,
    #    args=predictor.args
    #)
    yolo_strategy = yolo_strategy(
        model=model.predictor.model if 'v8' in str(args['yolo_model']) else args['yolo_model'],
        device=predictor.device,
        args=predictor.args
    )
    model = Yolo(yolo_strategy)

    # Var for timer, for counting a FPS
    timePerFrameArray = numpy.array([], dtype=float)

    # Temp vector of detected objects
    # Detected objects in currect picture frame from all perspective
    tempDetectedObjectsUnsorted = []

    # All sorted/paired detected objects
    #tempDetectedObjectsSorted = []

    for frame_idx, batch in enumerate(predictor.dataset):
        #print(f'frame index: {frame_idx}')
        #print(f'batch: {batch}')
        #print(frame_idx)
        kiactm = cv2.TickMeter()
        kiactm.start()

        # Start timer
        start = timer()
        predictor.run_callbacks('on_predict_batch_start')
        predictor.batch = batch
        path, im0s, vid_cap, s = batch

        n = len(im0s)
        predictor.results = [None] * n
        ### ### ###kiactm = cv2.TickMeter()
        ### ### ###kiactm.start()

        # Preprocess
        with predictor.profilers[0]:
            im = predictor.preprocess(im0s)
        ### ### ###kiactm.stop()
        ### ### ###kiacprint("Preprocess: ", tm.getTimeMilli())
        ### ### ###kiactm.reset()

        ### ### ###kiactm.start()
        # Inference
        with predictor.profilers[1]:
            preds = model.inference(im=im)

        ### ### ###kiactm.stop()
        ### ### ###kiacprint("Inference: ", tm.getTimeMilli())
        ### ### ###kiactm.reset()

        ### ### ###kiactm.start()
        # Postprocess moved to MultiYolo
        with predictor.profilers[2]:
            predictor.results = model.postprocess(path, preds, im, im0s, predictor)
        predictor.run_callbacks('on_predict_postprocess_end')

        # Visualize, save, write results
        n = len(im0s)
        ### ### ###kiactm.stop()
        ### ### ###kiacprint("Postprocess: ", tm.getTimeMilli())
        ### ### ###kiactm.reset()

        #print(f'Batch: {batch}')
        #print(f'n: {n}')

        # !!!
        # Toto iteruje cez viacero obrazkov, respektive v pripade kedy do modelu NN davame viacero vstupov na raz (viac bacthov)
        # !!!
        ### ### ###kiactm.start()
        for i in range(n):
            if predictor.dataset.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)
            #print(f'Index: {i}, Path: {p}')

            with predictor.profilers[3]:
                # get raw bboxes tensor
                dets = predictor.results[i].boxes.data
                # get tracker predictions
                predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach().numpy(), im0)
            predictor.results[i].speed = {
                'preprocess': predictor.profilers[0].dt * 1E3 / n,
                'inference': predictor.profilers[1].dt * 1E3 / n,
                'postprocess': predictor.profilers[2].dt * 1E3 / n,
                'tracking': predictor.profilers[3].dt * 1E3 / n
            }

            # filter boxes masks and pose results by tracking results
            model.filter_results(i, predictor)
            # overwrite bbox results with tracker predictions
            model.overwrite_results(i, im0.shape[:2], predictor)
            
            # https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box
            isMultiCameraTracking = True
            if isMultiCameraTracking:
                #results = model(source=..., stream=True)  # generator of Results objects
                results = predictor.results[i]
                for r in results:
                    boxes = r.boxes  # Boxes object for bbox outputs
                    #masks = r.masks  # Masks object for segment masks outputs
                    #probs = r.probs  # Class probabilities for classification outputs
                    for box in boxes:      

                        # Bbox class name                        
                        bbBoxNameIndex = box.cls
                        bbBoxName = r.names[int(bbBoxNameIndex)]

                        # Checking a allowedDetectedObjects (filter)
                        if not bbBoxName in allowedDetectedObjects:
                            continue

                        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                        bboxStartPoint = (int(b[0]), int(b[1]))
  
                        # Ending coordinate, here (220, 220)
                        # represents the bottom right corner of rectangle
                        bboxEndPoint = (int(b[2]), int(b[3]))
  
                        # Blue color in BGR
                        color = (255, 0, 0)
  
                        # Line thickness of 2 px
                        thickness = 2
  
                        # Using cv2.rectangle() method
                        # Draw a rectangle with blue line borders of thickness of 2 px
                        im0 = cv2.rectangle(im0, bboxStartPoint, bboxEndPoint, color, thickness)

                        # font
                        font = cv2.FONT_HERSHEY_SIMPLEX
  
                        # org
                        org = bboxStartPoint
  
                        # fontScale
                        fontScale = 1
   
                        # Blue color in BGR
                        color = (0, 0, 255)
  
                        # Line thickness of 2 px
                        thickness = 2
   
                        # Using cv2.putText() method
                        im0 = cv2.putText(im0, bbBoxName, org, font, fontScale, color, thickness, cv2.LINE_AA)

                        org = bboxEndPoint
                        color = (255, 255, 255)

                        if box.id is not None:
                            im0 = cv2.putText(im0, str(int(box.id[0])), org, font, fontScale, color, thickness, cv2.LINE_AA)

                        # b[0] -> x1
                        # b[1] -> y1
                        # b[2] -> x2
                        # b[3] -> y2
                        bboxCenterPointX = int(b[0]) + int((int(b[2]) - int(b[0]))/2)
                        bboxCenterPointY = int(b[1]) + int((int(b[3]) - int(b[1]))/2)
                        # Set perspective offset in y direction
                        bboxCenterPointY = bboxCenterPointY + int((int(b[3]) - int(b[1])) / 3)

                        
                        bboxWidth = int(b[2]) - int(b[0])
                        bboxHeight = int(b[3]) - int(b[1])

                        
                        # Draw center of detected object to camera perspective
                        cv2.circle(im0, (bboxCenterPointX, bboxCenterPointY), 2, (0, 0, 255), 2)

                        # multi camera, detected object pairing
                        # get x,y position in (digital twin/bird view) image frame
                        #point_to_ref = calibrations_functions.getPointInRefPlane((bboxCenterPointX, bboxCenterPointY), cd_vec[i+1].H)
                        point_to_ref = computePointInReferenceFrame(bboxCenterPointX, bboxCenterPointY, i)

                        #searchingAreaOffsetX = 84
                        #searchingAreaOffsetY = 48
                        searchingAreaOffsetX = 48
                        searchingAreaOffsetY = 48

                        if (i==0):
                            cv2.circle(birdViewFrame, point_to_ref, 2, (255, 0, 0), 2)
                            ###cv2.rectangle(birdViewFrame, (int(point_to_ref[0]-searchingAreaOffsetX), int(point_to_ref[1]-searchingAreaOffsetY)), (int(point_to_ref[0]+searchingAreaOffsetX), int(point_to_ref[1]+searchingAreaOffsetY)), (255, 0, 0), 1)
                            #cv2.drawMarker(birdViewFrame, point_to_ref, color=(255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                        elif (i==1):
                            cv2.circle(birdViewFrame, point_to_ref, 2, (0, 255, 0), 2)
                            ###cv2.rectangle(birdViewFrame, (int(point_to_ref[0]-searchingAreaOffsetX), int(point_to_ref[1]-searchingAreaOffsetY)), (int(point_to_ref[0]+searchingAreaOffsetX), int(point_to_ref[1]+searchingAreaOffsetY)), (0, 255, 0), 1)
                            #cv2.drawMarker(birdViewFrame, point_to_ref, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                        #elif (i==2):
                        #    cv2.circle(birdViewFrame, point_to_ref, 2, (0, 0, 255), 2)
                        #elif (i==3):
                        #    cv2.circle(birdViewFrame, point_to_ref, 2, (255, 255, 0), 2)

                        # Current detected object
                        #DetectedObject(id, className, x, y, width, height, detectedTime, gpsPosition)


                        #searchingAreaOffsetY = searchingAreaOffsetX

                        #cv2.rectangle(birdViewFrame, (int(point_to_ref[0]-searchingAreaOffsetY), int(point_to_ref[1]-searchingAreaOffsetY)), (int(point_to_ref[0]+searchingAreaOffsetY), int(point_to_ref[1]+searchingAreaOffsetY)), (255,0,0), 4)
                        
                        latitude, longitude = pixel_to_gps(point_to_ref[0], point_to_ref[1])
                        tempDetectedObject = DetectedObject(-1, bbBoxName, point_to_ref[0], point_to_ref[1], 25, 25, time.perf_counter(), (latitude, longitude))                       

                        # Check, if it iterate over last video/stream source
                        # Prebehne vsetky iteracie, zdroj po zdroji... Az na poslednej iteracii, tam iteruje posledny zdroj a rovno paruje s predchazdajucimi
                        if(len(tempDetectedObjectsUnsorted) > 0):
                            centerX, centerY = tempDetectedObject.getRectCenter()
                            #objCenterPoint = (centerX, centerY + (height * 0.5))
                            #cv2.circle(birdViewFrame, (int(centerX), int(centerY)), 2, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2)
                            #cv2.rectangle(birdViewFrame, (int(centerX-64), int(centerY-32)), (int(centerX+64), int(centerY+32)), (tempDetectedObject.blue, tempDetectedObject.green, tempDetectedObject.red), 4)

                            if(len(tempDetectedObjectsUnsorted) > 0 and tempDetectedObject is not None):
                                
                                # Temp store for all similar detected objects, array of indexes
                                similarDetectedObjectIndexes = []

                                # Temp store for all similar detected objects, array of positions
                                #similarDetectedObjectCenterPositions = []

                                for index, forDetectedObject in enumerate(tempDetectedObjectsUnsorted):
                                    forDetectedObjectCenterX, forDetectedObjectCenterY = forDetectedObject.getRectCenter()
                                    # Rect(x, y, width, height)
                                    dObjectRect = Rect(forDetectedObjectCenterX-searchingAreaOffsetX, forDetectedObjectCenterY-searchingAreaOffsetY, searchingAreaOffsetX, searchingAreaOffsetY)
                                    if dObjectRect.contains(centerX, centerY):
                                        similarDetectedObjectIndexes.append(index)

                                if len(similarDetectedObjectIndexes) > 0:
                                    similarDetectedObjectIndexesSorted = sorted(similarDetectedObjectIndexes, reverse=True)
                                    similarDetectedObjectPositionsCenterX = []
                                    similarDetectedObjectPositionsCenterY = []
                                    for index in similarDetectedObjectIndexesSorted:
                                        similarDetObCenterX, similarDetObCenterY = tempDetectedObjectsUnsorted[index].getRectCenter()
                                        similarDetectedObjectPositionsCenterX.append(similarDetObCenterX)
                                        similarDetectedObjectPositionsCenterY.append(similarDetObCenterY)
                                        del tempDetectedObjectsUnsorted[index]
                                        #removeDetectedObjectIndex(index)

                                    similarDetectedObjectPositionsCenterX.append(centerX)
                                    similarDetectedObjectPositionsCenterY.append(centerY)
                                    averageDetObPositionsCenterX = int(sum(similarDetectedObjectPositionsCenterX) / len(similarDetectedObjectPositionsCenterX))
                                    averageDetObPositionsCenterY = int(sum(similarDetectedObjectPositionsCenterY) / len(similarDetectedObjectPositionsCenterY))

                                    latitude, longitude = pixel_to_gps(averageDetObPositionsCenterX, averageDetObPositionsCenterY)
                                    resultingDetectedObject = DetectedObject(-1, bbBoxName, averageDetObPositionsCenterX, averageDetObPositionsCenterY, 25, 25, time.perf_counter(), (latitude, longitude))

                                    # Teraz len priradenie s predchadzajucimi detekciami, to znamena aktualizacia detekovanych objektov novymi
                                    if len(detectedObjects) > 0:
                                        nearestDistance = 99999
                                        nearestDetectedObjectIndex = -1
                                        hasSimilarObjectFound = False

                                        for index, forDetectedObject in enumerate(detectedObjects):
                                            forDetectedObjectCenterX, forDetectedObjectCenterY = forDetectedObject.getRectCenter()
                                            # Rect(x, y, width, height)
                                            #dObjectRect = Rect(forDetectedObjectCenterX-searchingAreaOffsetX, forDetectedObjectCenterY-searchingAreaOffsetY, searchingAreaOffsetX, searchingAreaOffsetY)
                                            dObjectRect = Rect(forDetectedObjectCenterX-searchingAreaOffsetX, forDetectedObjectCenterY-searchingAreaOffsetY, searchingAreaOffsetX*2, searchingAreaOffsetY*2)
                                            resultingDetectedObjectCenterX = resultingDetectedObject.getRectCenter()[0]
                                            resultingDetectedObjectCenterY = resultingDetectedObject.getRectCenter()[1]

                                            if dObjectRect.contains(resultingDetectedObjectCenterX, resultingDetectedObjectCenterY):
                                                # x difference
                                                xDifference = abs(forDetectedObject.getObjectTrajectory()[len(forDetectedObject.getObjectTrajectory())-1][0] - resultingDetectedObjectCenterX)
                                                # x difference
                                                yDifference = abs(forDetectedObject.getObjectTrajectory()[len(forDetectedObject.getObjectTrajectory())-1][1] - resultingDetectedObjectCenterY)
                                                # Distance
                                                distance = math.sqrt(math.pow(xDifference,2) + math.pow(yDifference,2))
                                                #
                                                if distance < nearestDistance:
                                                    nearestDetectedObjectIndex = index
                                                
                                                hasSimilarObjectFound = True
                            
                                        if hasSimilarObjectFound == True:
                                            latitude, longitude = pixel_to_gps(resultingDetectedObjectCenterX, resultingDetectedObjectCenterY)
                                            detectedObjects[nearestDetectedObjectIndex].updateObject(bbBoxName, resultingDetectedObjectCenterX, resultingDetectedObjectCenterY, time.perf_counter(), (latitude, longitude))                               
                                            detectedObjects[nearestDetectedObjectIndex].resetTTL()
                                        else:
                                            detectedObjectIndexesCounter = detectedObjectIndexesCounter + 1
                                            detectedObjects.append(DetectedObject(detectedObjectIndexesCounter, resultingDetectedObject.getClassName(), resultingDetectedObject.getX(), resultingDetectedObject.getY(), 25, 25, resultingDetectedObject.getDetectedTime(), resultingDetectedObject.getGpsPosition()))   
                                            del resultingDetectedObject
                                    else:
                                        detectedObjectIndexesCounter = detectedObjectIndexesCounter + 1
                                        detectedObjects.append(DetectedObject(detectedObjectIndexesCounter, resultingDetectedObject.getClassName(), resultingDetectedObject.getX(), resultingDetectedObject.getY(), 25, 25, resultingDetectedObject.getDetectedTime(), resultingDetectedObject.getGpsPosition()))
                                        del resultingDetectedObject
                                else:
                                    # Opat len priradenie s predchadzajucimi detekciami, to znamena aktualizacia detekovanych objektov novymi
                                    if len(detectedObjects) > 0:
                                        nearestDistance = 99999
                                        nearestDetectedObjectIndex = -1
                                        hasSimilarObjectFound = False

                                        for index, forDetectedObject in enumerate(detectedObjects):
                                            forDetectedObjectCenterX, forDetectedObjectCenterY = forDetectedObject.getRectCenter()
                                            # Rect(x, y, width, height)
                                            #dObjectRect = Rect(forDetectedObjectCenterX-searchingAreaOffsetX, forDetectedObjectCenterY-searchingAreaOffsetY, searchingAreaOffsetX, searchingAreaOffsetY)
                                            dObjectRect = Rect(forDetectedObjectCenterX-searchingAreaOffsetX, forDetectedObjectCenterY-searchingAreaOffsetY, searchingAreaOffsetX*2, searchingAreaOffsetY*2)
                                            tempDetectedObjectCenterX = tempDetectedObject.getRectCenter()[0]
                                            tempDetectedObjectCenterY = tempDetectedObject.getRectCenter()[1]

                                            if dObjectRect.contains(tempDetectedObjectCenterX, tempDetectedObjectCenterY):
                                                # x difference
                                                xDifference = abs(forDetectedObject.getObjectTrajectory()[len(forDetectedObject.getObjectTrajectory())-1][0] - tempDetectedObjectCenterX)
                                                # x difference
                                                yDifference = abs(forDetectedObject.getObjectTrajectory()[len(forDetectedObject.getObjectTrajectory())-1][1] - tempDetectedObjectCenterY)
                                                # Distance
                                                distance = math.sqrt(math.pow(xDifference,2) + math.pow(yDifference,2))
                                                #
                                                if distance < nearestDistance:
                                                    nearestDetectedObjectIndex = index
                                                
                                                hasSimilarObjectFound = True
                            
                                        if hasSimilarObjectFound == True:
                                            latitude, longitude = pixel_to_gps(tempDetectedObjectCenterX, tempDetectedObjectCenterY)
                                            detectedObjects[nearestDetectedObjectIndex].updateObject(bbBoxName, tempDetectedObjectCenterX, tempDetectedObjectCenterY, time.perf_counter(), (latitude, longitude))                               
                                            detectedObjects[nearestDetectedObjectIndex].resetTTL()
                                        else:
                                            detectedObjectIndexesCounter = detectedObjectIndexesCounter + 1
                                            detectedObjects.append(DetectedObject(detectedObjectIndexesCounter, tempDetectedObject.getClassName(), tempDetectedObject.getX(), tempDetectedObject.getY(), 25, 25, tempDetectedObject.getDetectedTime(), tempDetectedObject.getGpsPosition()))   
                                            del tempDetectedObject
                                    else:
                                        detectedObjectIndexesCounter = detectedObjectIndexesCounter + 1
                                        detectedObjects.append(DetectedObject(detectedObjectIndexesCounter, tempDetectedObject.getClassName(), tempDetectedObject.getX(), tempDetectedObject.getY(), 25, 25, tempDetectedObject.getDetectedTime(), tempDetectedObject.getGpsPosition()))      
                                        del tempDetectedObject
                        else:
                            tempDetectedObjectsUnsorted.append(tempDetectedObject)
                            del tempDetectedObject

                                 
            # write inference results to a file or directory
            if (predictor.args.verbose or predictor.args.save or
               predictor.args.save_txt or predictor.args.show or
               predictor.args.save_id_crops):

                s += predictor.write_results(i, predictor.results, (p, im, im0))
                predictor.txt_path = Path(predictor.txt_path)

                ########kiac
                #for r in results:
                #    boxes = r.boxes  # Boxes object for bbox outputs
                #print("########################################################")
                #https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box
                #
                ###print(predictor.results[i].boxes.xywh)
                #print(predictor.results[i].boxes[0].xyxy)
                ###print(predictor.results[i].boxes.cls)
                #print(predictor.results[i].boxes.id)
                #print(model.names[int(predictor.results[i].boxes.cls)])
                ###print(predictor.results[i].boxes.cls)
                #print(predictor.results[i].names[0])

                #model.names[int(c)]
                
                #print(predictor.results[i].probs)
                #print("########################################################")
                #cv2.imshow("im0", im0)

                # write MOT specific results
                if predictor.args.source.endswith(VID_FORMATS):
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
                # mot txt called the same as the parent name to perform inference on
                elif 'MOT16' or 'MOT17' or 'MOT20' in predictor.args.source:
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.parent.name
                # mot txt called the same as the parent name to perform inference on
                else:              
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name

                if predictor.tracker_outputs[i].size != 0 and predictor.args.save_mot:
                    write_MOT_results(
                        predictor.MOT_txt_path,
                        predictor.results[i],
                        frame_idx,
                        i,
                    )

                if predictor.args.save_id_crops:
                    for d in predictor.results[i].boxes:
                        save_one_box(
                            d.xyxy,
                            im0.copy(),
                            file=(predictor.save_dir / 'crops' /
                                  str(int(d.cls.cpu().numpy().item())) /
                                  str(int(d.id.cpu().numpy().item())) / f'{p.stem}.jpg'),
                            BGR=True
                        )

            # display an image in a window using OpenCV imshow()
            if predictor.args.show and predictor.plotted_img is not None:
                predictor.show(p.parent)

            # save video predictions
            if predictor.args.save and predictor.plotted_img is not None:
                predictor.save_preds(vid_cap, i, str(predictor.save_dir / p.name))

            isShowImageWithDNNOutputs = True
            if isShowImageWithDNNOutputs == True:
                im0FrameForView = cv2.resize(im0, (int(im0.shape[1]/2), int(im0.shape[0]/2)), interpolation = cv2.INTER_AREA)
                #kiac
                #cv2.imshow("Birdvieweee", birdViewFrame)
                #Zobrazenie perspektivy kamier
                #Zobrazenie kamier
                cv2.imshow(f'{p}', im0FrameForView)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()

                #    exit()
        ### ### ###kiactm.stop()
        ### ### ###kiacprint("Tracking: ", tm.getTimeMilli())
        ### ### ###kiactm.reset()

        ###########################################################
        ObjectListJson["Objects"] = []
        canDrawDetectedObjectToBirdView = True
        if canDrawDetectedObjectToBirdView == True:
            #showDetectedObjectIndexes()
            #birdViewPath = "D:\\DNN\\yunex_traffic_dnn_py\\cameraViews\\crossroadX_birdView.png"
            #birdViewPath = topDownImageFrame
            birdViewFrameTesting = cv2.imread(birdViewPath, cv2.IMREAD_COLOR)
            for index, forDetectedObject in enumerate(detectedObjects):
                
                detectedObjects[index].decreaseTTL()
                if detectedObjects[index].getTTL() <= 0:
                    detectedObjects.pop(index)
                    #removeDetectedObjectIndex(index)
                    continue           

                x, y, width, height = forDetectedObject.getRect()

                centerX, centerY = forDetectedObject.getRectCenter()
                #objCenterPoint = (centerX, centerY + (height * 0.5))
                cv2.circle(birdViewFrameTesting, (int(centerX), int(centerY)), 2, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2)
                ###cv2.rectangle(birdViewFrameTesting, (int(centerX-42), int(centerY-32)), (int(centerX+42), int(centerY+32)), (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 4)            
                cv2.rectangle(birdViewFrameTesting, (int(centerX-int(searchingAreaOffsetX/4)), int(centerY-int(searchingAreaOffsetX/4))), (int(centerX+int(searchingAreaOffsetX/4)), int(centerY+int(searchingAreaOffsetX/4))), (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 4)            
                cv2.putText(birdViewFrameTesting, forDetectedObject.getClassName(), (int(centerX), int(centerY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2, cv2.LINE_AA)
                cv2.putText(birdViewFrameTesting, str(forDetectedObject.getId()), (int(centerX), int(centerY+24)), cv2.FONT_HERSHEY_SIMPLEX, 1, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2, cv2.LINE_AA)

                #currentObjectSpeed = "{:.2f}".format(forDetectedObject.getSpeed())
                currentObjectSpeed = forDetectedObject.getSpeed()
                #objectSpeedValueString = "{} km/h".format(str(forDetectedObject.getSpeed()))
                objectSpeedValueString = "NaN"
                objectSpeedValueStringForView = 0

                if currentObjectSpeed <= 2.5:
                    objectSpeedValueString = "{:.2f}".format(0)
                    objectSpeedValueStringForView = 0
                else:
                    objectSpeedValueString = "{:.2f}".format(forDetectedObject.getSpeed())
                    objectSpeedValueStringForView = forDetectedObject.getSpeed() * 3.6

                objectSpeedValueStringForView = "{:.2f}".format(objectSpeedValueStringForView)
                cv2.putText(birdViewFrameTesting, objectSpeedValueStringForView + " km/h", (int(centerX), int(centerY+48)), cv2.FONT_HERSHEY_SIMPLEX, 1, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2, cv2.LINE_AA)
                
                # Detected object counting
                if not forDetectedObject.getId() in detectedObjectIdInCounter:
                    detectedObjectIdInCounter.append(forDetectedObject.getId())                 
                    if forDetectedObject.getClassName() == "car":
                        global countCar
                        countCar = countCar + 1
                    if forDetectedObject.getClassName() == "bus":
                        global countBus
                        countBus = countBus + 1
                    if forDetectedObject.getClassName() == "truck":
                        global countTruck
                        countTruck = countTruck + 1
                    if forDetectedObject.getClassName() == "train/tram":
                        global countTrainTram
                        countTrainTram = countTrainTram + 1

                #pointInReferenceFrame = computePointInReferenceFrame(centerX, centerY, cd_vec[1])
                #cv2.circle(cd_vec[0].img, pointInReferenceFrame, 2, (0, 0, 255), 2)
                #cv2.circle(birdViewFrameTesting, (int(x + (width/2)), int(y + (height/2))), 2, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2)
                #cv2.rectangle(birdViewFrameTesting, (int(pointInReferenceFrame[0]-10), int(pointInReferenceFrame[1]-10)), (int(pointInReferenceFrame[0]+10), int(pointInReferenceFrame[1]+10)), (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 3)
            
                objTrajectory = forDetectedObject.getObjectTrajectory()
                for objPoint in objTrajectory:
                    #pointInReferenceFrame = computePointInReferenceFrame(objPoint[0], objPoint[1], cd_vec[1])
                    cv2.circle(birdViewFrameTesting, (int(objPoint[0]), int(objPoint[1])), 2, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2)

                ###########################################################
                ######################TESTTTTTT############################
                ###########################################################
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                #print(current_time)
                #print("[{}] main".format(current_time))

                #print(ObjectListJson)
                objectJson = {
                "Category": str(forDetectedObject.getClassName()),
                "Color": None,
                "Id": str(forDetectedObject.getId()),
                    "LastState": {
                        "BoundingBox": [None, None, None, None],
                        "MapAcceleration": None,
                        "MapPosition": [None, None],
                        "MapSpeed": objectSpeedValueString,
                        "MapSpeedAngle": None,
                        "SensorPosition": [None, None],
                        "Timestamp": str(int(time.time())*1000),
                        "WGS84Position": [float(forDetectedObject.getGpsPosition()[1]), float(forDetectedObject.getGpsPosition()[0])]
                    }
                }
                ObjectListJson["Objects"].append(objectJson)

                ###########################################################
                ######################TESTTTTTT############################
                ###########################################################

            birdViewFrameTesting = cv2.resize(birdViewFrameTesting, (int(birdViewFrameTesting.shape[1]/2), int(birdViewFrameTesting.shape[0]/2)), interpolation = cv2.INTER_AREA)
            cv2.imshow("TopDownView", birdViewFrameTesting)

            #global globalFrame
            #globalFrame = birdViewFrame
        #detectedObjects = []
        ###########################################################

        kiactm.stop()
        #print("Inference: ", kiactm.getTimeMilli())
        kiactm.reset()

        isShowBirdView = True
        if isShowBirdView == True:
            birdViewFrameForView = cv2.resize(birdViewFrame, (int(birdViewFrame.shape[1]/2), int(birdViewFrame.shape[0]/2)), interpolation = cv2.INTER_AREA)
            cv2.imshow("[DEBUG] TopDownView", birdViewFrameForView)

        # Wait for a key press and handle it
        key = cv2.waitKey(1) & 0xFF
        # Exit the loop if 'c' is pressed (ROI selection completed)
        if key == ord('r'):
            #birdViewPath = "D:\\DNN\\yunex_traffic_dnn_py\\cameraViews\\crossroadX_birdView.png"
            birdViewPath = topDownImageFrame
            birdViewFrame = cv2.imread(birdViewPath, cv2.IMREAD_COLOR)

        # Exit the loop if 'q' or Esc is pressed (cancel ROI selection)
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            stop_websocket_thread = True
            #exit()
            break

        predictor.run_callbacks('on_predict_batch_end')

        if len(timePerFrameArray) > 30:
            timePerFrameArray = numpy.delete(timePerFrameArray, 0)
            timePerFrameArray = numpy.append (timePerFrameArray, (timer()-start))
        else:   
            timePerFrameArray = numpy.append (timePerFrameArray, (timer()-start))

        #kiac#print("Current FPS: {:.2f}".format(1/(timer()-start)))
        #kiac#print("Average FPS: {:.2f}".format(1/((numpy.sum(timePerFrameArray, dtype = numpy.float32))/len(timePerFrameArray))))
        #kiac#print("########################################################")

        #cv2.putText(im0, "Current FPS: {:.2f}".format(1/(timer()-start)), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1, cv2.LINE_AA)
        #cv2.putText(im0, "Average FPS: {:.2f}".format(1/((numpy.sum(timePerFrameArray, dtype = numpy.float32))/len(timePerFrameArray))), (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1, cv2.LINE_AA)

        # print time (inference-only)
        showInferenceInfo = False
        if predictor.args.verbose and showInferenceInfo == True:
            s_t = f'YOLO {predictor.profilers[1].dt * 1E3:.1f}ms, TRACKING {predictor.profilers[3].dt * 1E3:.1f}ms'
            LOGGER.info(f'{s}{s_t}')

    # Release assets
    if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
        predictor.vid_writer[-1].release()  # release final video writer

    # Print results
    if predictor.args.verbose and predictor.seen:
        t = tuple(x.t / predictor.seen * 1E3 for x in predictor.profilers)  # speeds per image
        LOGGER.info(f'Speed: %.1fms preproc, %.1fms inference, %.1fms postproc, %.1fms tracking per image at shape '
                    f'{(1, 3, *predictor.args.imgsz)}' % t)
    if predictor.args.save or predictor.args.save_txt or predictor.args.save_crop:
        nl = len(list(predictor.save_dir.glob('labels/*.txt')))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}" if predictor.args.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    predictor.run_callbacks('on_predict_end')


def parse_opt():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', action="help", help="Show this help message and exit")
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'best.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=EXAMPLES / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='hide labels when show')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', action='store_true',
                        help='not mix up classes when tracking')

    opt = parser.parse_args()
    return opt
    
def loadCalibrationParametersFromXML(filePathNameToLoad):
    global cameraMatrix, distCoeffs
    print("[APP]: Loading camera parameters.")
    fs = cv2.FileStorage(filePathNameToLoad, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print("[APP]: Failed to open camera calibration parameters.")
        return None, None, None, None

    cameraMatrix = fs.getNode("cameraMatrix").mat()
    distCoeffs = fs.getNode("distCoeffs").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    fs.release()  # Close the file

    print("[APP]: Camera parameters successfully loaded.")

def loadHomographyMatricsFromJson(config_file_path_name_to_load):
    global homographyMatrix
    # Open and parse the JSON file
    with open(config_file_path_name_to_load, 'r') as file:
        data = json.load(file)

    # Iterate over each homography matrix in the JSON data
    for mat_json in data["homographyMatrix"]:
        # Create a 3x3 numpy array for the homography matrix
        mat = np.zeros((3, 3), dtype=np.float64)

        for i in range(3):
            for j in range(3):
                mat[i, j] = mat_json[i][j]

        # Append the numpy array to the list
        homographyMatrix.append(mat)

def loadGpsCoordinatesRelationFromJson(json_file_path):
    gps_coords_temp = []
    pixel_coords_temp = []

    # Open and parse the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Iterate over each entry in the 'topDownViewGpsCoordinates' list
    for entry in data["topDownViewGpsCoordinates"]:
        x, y = entry["x"], entry["y"]
        lat, long = entry["lat"], entry["long"]

        # Append (x, y) to gps_coords and (lat, long) to pixel_coords
        gps_coords_temp.append((x, y))
        pixel_coords_temp.append((float(lat), float(long)))

    # Convert to NumPy arrays
    gps_coords = np.array(gps_coords_temp)
    pixel_coords = np.array(pixel_coords_temp)

if __name__ == "__main__":
    opt = parse_opt()

    loadCalibrationParametersFromXML("cameraIntrinsicParameters_dulov.xml")
    loadHomographyMatricsFromJson("homographyConfig_dulov.json")

    loadGpsCoordinatesRelationFromJson("homographyConfig_UE5.json")

    #print(cameraMatrix)
    #print(homographyMatrix[0])
    #print(homographyMatrix[1])
    #print(gps_coords)
    #print(pixel_coords)

    # Create a thread for WebSocket communication
    ########videoStream_thread = threading.Thread(target=runVideoStream_thread)
    ########videoStream_thread.start()

    #horizontalChart_thread = threading.Thread(target=runHorizontalChart_thread)
    #horizontalChart_thread.start()

    ########sendingStatistics_thread = threading.Thread(target=runSendingStatistics_thread)
    ########sendingStatistics_thread.start()

    runProcessing(vars(opt))
    #websocket_thread.join()  # Wait for the thread to finish
    #t1.join()
    #t2.join()
