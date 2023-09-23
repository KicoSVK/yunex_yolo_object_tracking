# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

import argparse
from pathlib import Path
import sys
import subprocess
import threading
import logging

import cv2
import torch
import numpy
from timeit import default_timer as timer
import time
import random
import math

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

class CameraData:
    def __init__(self):
        self.img = None
        self.camera_idx = 0
        self.window_name = ""
        self.win_pos = None
        self.H = None
        self.K = None
        self.D = None
        self.P = None

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

def computePointInReferenceFrame(x, y, param):
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
    def __init__(self, id, className, x, y, width, height, detectedTime, realWorldPosition):
        self.id = id
        self.className = className
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.ttl = 28
        self.objectTrajectory = []
        self.objectTrajectory.append((x, y))
        self.centerX = x + (width/2)
        self.centerY = y + (height/2)
        self.detectedTime = detectedTime
        self.realWorldPosition = realWorldPosition
        self.speed = 0

        self.red = random.randrange(0, 255)
        self.green = random.randrange(0, 255)
        self.blue = random.randrange(0, 255)

    def draw(self, image):
            cv2.rectangle(image, (self.x, self.y), (self.x + self.width, self.y + self.height), (0, 255, 0), 2)

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

    def setSpeed(self, speed):
        self.speed = speed

    def getSpeed(self):
        #self.speed = random.uniform(25.1, 29.9)
        #return (self.speed/10)/4
        return 0

    def decreaseTTL(self):
        self.ttl = self.ttl - 1

    def resetTTL(self):
        self.ttl = 28

    def getTTL(self):
        return self.ttl

    def updateObject(self, className, x, y, detectedTime, realWorldPosition):
        self.className = className
        self.x = x
        self.y = y
        self.centerX = x + (self.width/2)
        self.centerY = y + (self.height/2)
        self.objectTrajectory.append((self.centerX, self.centerY))
        self.detectedTime = detectedTime
        self.realWorldPosition = realWorldPosition

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

    cd_1 = CameraData()
    cd_2 = CameraData()
    cd_3 = CameraData()
    cd_4 = CameraData()
    cd_vec = []

    # load imgs, idxs etc...
    cd_ref.camera_idx = 0
    cd_ref.img = cv2.imread("cameraViews/crossroadX_birdView.png")
    cd_ref.window_name = cam_ref_win_name
    cd_ref.win_pos = (500, 500)
    cd_vec.append(cd_ref)

    cd_1.camera_idx = 1
    cd_1.img = cv2.imread("cameraViews/crossroadX_cam_01.png")
    cd_1.window_name = "cam_1"
    cd_1.win_pos = (0, 0)
    cd_vec.append(cd_1)

    cd_2.camera_idx = 2
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

    birdViewPath = "D:\\DNN\\\yunex_yolo_object_tracking\\examples\\cameraViews\\crossroadX_birdView.png"
    birdViewFrame = cv2.imread(birdViewPath, cv2.IMREAD_COLOR)

    model = YOLO(args['yolo_model'] if 'v8' in str(args['yolo_model']) else 'yolov8n')
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)

    # detected objects
    allowedDetectedObjects = ["person", "car", "truck", "bus", "tram/train", "bicycle", "motorcycle"]
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

    for frame_idx, batch in enumerate(predictor.dataset):
        #print(f'frame index: {frame_idx}')
        #print(f'batch: {batch}')

        # Start timer
        start = timer()
        predictor.run_callbacks('on_predict_batch_start')
        predictor.batch = batch
        path, im0s, vid_cap, s = batch

        # Temp vector of detected objects
        # Detected objects in currect picture frame from all perspective
        tempDetectedObjectsUnsorted = []

        # All sorted/paired detected objects
        tempDetectedObjectsSorted = []

        n = len(im0s)
        predictor.results = [None] * n

        # Preprocess
        with predictor.profilers[0]:
            im = predictor.preprocess(im0s)

        # Inference
        with predictor.profilers[1]:
            preds = model.inference(im=im)

        # Postprocess moved to MultiYolo
        with predictor.profilers[2]:
            predictor.results = model.postprocess(path, preds, im, im0s, predictor)
        predictor.run_callbacks('on_predict_postprocess_end')

        # Visualize, save, write results
        n = len(im0s)

        #print(f'Batch: {batch}')
        #print(f'n: {n}')

        # !!!
        # Toto iteruje cez viacero obrazkov, respektive v pripade kedy do modelu NN davame viacero vstupov na raz (viac bacthov)
        # !!!
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
                        point_to_ref = calibrations_functions.getPointInRefPlane((bboxCenterPointX, bboxCenterPointY), cd_vec[i+1].H)

                        if (i==0):
                            cv2.circle(birdViewFrame, point_to_ref, 2, (255, 0, 0), 2)
                            #cv2.drawMarker(birdViewFrame, point_to_ref, color=(255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                        elif (i==1):
                            cv2.circle(birdViewFrame, point_to_ref, 2, (0, 255, 0), 2)
                            #cv2.drawMarker(birdViewFrame, point_to_ref, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                        #elif (i==2):
                        #    cv2.circle(birdViewFrame, point_to_ref, 2, (0, 0, 255), 2)
                        #elif (i==3):
                        #    cv2.circle(birdViewFrame, point_to_ref, 2, (255, 255, 0), 2)

                        # Current detected object
                        tempDetectedObject = DetectedObject(len(detectedObjects), bbBoxName, point_to_ref[0], point_to_ref[1], 25, 25, 0.0, (0, 0))
                        
                        searchingAreaOffsetX = 55
                        searchingAreaOffsetY = 55

                        # Check, if it iterate over last video/stream source
                        # Prebehne vsetky iteracie, zdroj po zdroji... Az na poslednej iteracii, tam iteruje posledny zdroj a rovno paruje s predchazdajucimi
                        if i == (n-1):
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
                                    dObjectRect = Rect(forDetectedObjectCenterX-84, forDetectedObjectCenterY-48, 168, 96)
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

                                    similarDetectedObjectPositionsCenterX.append(centerX)
                                    similarDetectedObjectPositionsCenterY.append(centerY)
                                    averageDetObPositionsCenterX = int(sum(similarDetectedObjectPositionsCenterX) / len(similarDetectedObjectPositionsCenterX))
                                    averageDetObPositionsCenterY = int(sum(similarDetectedObjectPositionsCenterY) / len(similarDetectedObjectPositionsCenterY))

                                    resultingDetectedObject = DetectedObject(len(detectedObjects), bbBoxName, averageDetObPositionsCenterX, averageDetObPositionsCenterY, 25, 25, 0.0, (0, 0))

                                    # Teraz len priradenie s predchadzajucimi detekciami, to znamena aktualizacia detekovanych objektov novymi
                                    if len(detectedObjects) > 0:
                                        nearestDistance = 99999
                                        nearestDetectedObjectIndex = -1
                                        hasSimilarObjectFound = False

                                        for index, forDetectedObject in enumerate(detectedObjects):
                                            forDetectedObjectCenterX, forDetectedObjectCenterY = forDetectedObject.getRectCenter()
                                            # Rect(x, y, width, height)
                                            dObjectRect = Rect(forDetectedObjectCenterX-84, forDetectedObjectCenterY-48, 168, 96)
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
                                            detectedObjects[nearestDetectedObjectIndex].updateObject(bbBoxName, resultingDetectedObjectCenterX, resultingDetectedObjectCenterY, 0.0, (0, 0))                               
                                            detectedObjects[nearestDetectedObjectIndex].resetTTL()
                                        else:
                                            detectedObjects.append(resultingDetectedObject)                                          
                                    else:
                                        detectedObjects.append(resultingDetectedObject)
                                else:
                                    # Opat len priradenie s predchadzajucimi detekciami, to znamena aktualizacia detekovanych objektov novymi
                                    if len(detectedObjects) > 0:
                                        nearestDistance = 99999
                                        nearestDetectedObjectIndex = -1
                                        hasSimilarObjectFound = False

                                        for index, forDetectedObject in enumerate(detectedObjects):
                                            forDetectedObjectCenterX, forDetectedObjectCenterY = forDetectedObject.getRectCenter()
                                            # Rect(x, y, width, height)
                                            dObjectRect = Rect(forDetectedObjectCenterX-84, forDetectedObjectCenterY-48, 168, 96)
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
                                            detectedObjects[nearestDetectedObjectIndex].updateObject(bbBoxName, tempDetectedObjectCenterX, tempDetectedObjectCenterY, 0.0, (0, 0))                               
                                            detectedObjects[nearestDetectedObjectIndex].resetTTL()
                                        else:
                                            detectedObjects.append(tempDetectedObject)
                                    else:
                                        detectedObjects.append(tempDetectedObject)                                                                  
                        else:
                            tempDetectedObjectsUnsorted.append(tempDetectedObject)

                                 
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
                cv2.imshow(f'{p}', im0FrameForView)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    exit()


        ###########################################################
            canDrawDetectedObjectToBirdView = True
            if canDrawDetectedObjectToBirdView == True:
                birdViewPath = "D:\\DNN\\yunex_traffic_dnn_py\\cameraViews\\crossroadX_birdView.png"
                birdViewFrame = cv2.imread(birdViewPath, cv2.IMREAD_COLOR)
                for index, forDetectedObject in enumerate(detectedObjects):
                
                    forDetectedObject.decreaseTTL()
                    if forDetectedObject.getTTL() <= 0:
                        detectedObjects.pop(index)
                        continue
                    

                    x, y, width, height = forDetectedObject.getRect()
                    centerX, centerY = forDetectedObject.getRectCenter()
                    #objCenterPoint = (centerX, centerY + (height * 0.5))
                    #cv2.circle(birdViewFrame, (int(centerX), int(centerY)), 2, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2)
                    cv2.rectangle(birdViewFrame, (int(centerX-42), int(centerY-32)), (int(centerX+42), int(centerY+32)), (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 4)            
                    cv2.putText(birdViewFrame, forDetectedObject.getClassName(), (int(centerX), int(centerY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2, cv2.LINE_AA)
                    cv2.putText(birdViewFrame, str(forDetectedObject.getId()), (int(centerX), int(centerY+24)), cv2.FONT_HERSHEY_SIMPLEX, 1, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2, cv2.LINE_AA)

                    speedQ = "{:.2f}".format(forDetectedObject.getSpeed())
                    #objectSpeedValueString = "{} km/h".format(str(forDetectedObject.getSpeed()))
                    objectSpeedValueString = "{} km/h".format(str(speedQ))
                    #cv2.putText(birdViewFrame, objectSpeedValueString, (int(centerX), int(centerY+48)), cv2.FONT_HERSHEY_SIMPLEX, 1, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2, cv2.LINE_AA)

                    #pointInReferenceFrame = computePointInReferenceFrame(centerX, centerY, cd_vec[1])
                    #cv2.circle(cd_vec[0].img, pointInReferenceFrame, 2, (0, 0, 255), 2)
                    #cv2.circle(birdViewFrame, (int(x + (width/2)), int(y + (height/2))), 2, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2)
                    #cv2.rectangle(birdViewFrame, (int(pointInReferenceFrame[0]-10), int(pointInReferenceFrame[1]-10)), (int(pointInReferenceFrame[0]+10), int(pointInReferenceFrame[1]+10)), (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 3)
            
                    objTrajectory = forDetectedObject.getObjectTrajectory()
                    for objPoint in objTrajectory:
                        #pointInReferenceFrame = computePointInReferenceFrame(objPoint[0], objPoint[1], cd_vec[1])
                        cv2.circle(birdViewFrame, (int(objPoint[0]), int(objPoint[1])), 2, (forDetectedObject.blue, forDetectedObject.green, forDetectedObject.red), 2)

            #detectedObjects = []
        ###########################################################

        isShowBirdView = True
        if isShowBirdView == True:
            birdViewFrameForView = cv2.resize(birdViewFrame, (int(birdViewFrame.shape[1]/2), int(birdViewFrame.shape[0]/2)), interpolation = cv2.INTER_AREA)
            cv2.imshow("Birdview", birdViewFrameForView)

        # Wait for a key press and handle it
        key = cv2.waitKey(1) & 0xFF

        # Exit the loop if 'c' is pressed (ROI selection completed)
        if key == ord('r'):
            birdViewPath = "D:\\DNN\\yunex_traffic_dnn_py\\cameraViews\\crossroadX_birdView.png"
            birdViewFrame = cv2.imread(birdViewPath, cv2.IMREAD_COLOR)

        # Exit the loop if 'q' or Esc is pressed (cancel ROI selection)
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            exit()
            break

        predictor.run_callbacks('on_predict_batch_end')

        if len(timePerFrameArray) > 30:
            timePerFrameArray = numpy.delete(timePerFrameArray, 0)
            timePerFrameArray = numpy.append (timePerFrameArray, (timer()-start))
        else:   
            timePerFrameArray = numpy.append (timePerFrameArray, (timer()-start))

        print("Current FPS: {:.2f}".format(1/(timer()-start)))
        print("Average FPS: {:.2f}".format(1/((numpy.sum(timePerFrameArray, dtype = numpy.float32))/len(timePerFrameArray))))
        print("########################################################")

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
    


if __name__ == "__main__":
    opt = parse_opt()

    runProcessing(vars(opt))

    #t1.join()
    #t2.join()
