import asyncio
import websockets
import cv2
import base64
import threading

from datetime import datetime
import time
import json
import random
import numpy as np

iterateee = 0
globalFrame = None

countCar = 0
countBus = 0
countTruck = 0
countTrainTram = 0

killAsyncThreadForChar = False
stop_websocket_thread = False

detectedObjects = [1]

ObjectListJson = {
          "AnalyticsId": 0,
          "EvaluationTimestamp": str(int(time.time())*1000),
          "Failure": False,
          "FailureState": "None",
          "Objects": [],
          "supportedClassesForCounter": ["Car", "Bus", "Truck", "Train/Tram"],
          "totalObjectsCountForCounter": [],
          "Part": 1,
          "TotalParts": 1
        }

async def sendDataToHorizontalChart(websocket, path):
    #global stop_websocket_thread
    while not stop_websocket_thread:
        global countCar
        countCar = countCar + random.randint(0, 4)
        global countBus
        countBus = countBus + random.randint(0, 1)
        global countTruck
        countTruck = countTruck + random.randint(0, 1)
        global countTrainTram
        countTrainTram = countTrainTram + random.randint(0, 0)

        #data = {
        #    "supportedClassesForCounter": ["Car", "Bus", "Truck", "Train/Trammm"],
        #    "totalObjectsCountForCounter": [countCar, countBus, countTruck, countTrainTram]
        #}
        ObjectListJson["totalObjectsCountForCounter"] = [countCar, countBus, countTruck, countTrainTram]

        now = datetime.now()
        #current_time = now.strftime("async thread: %H:%M:%S")
        current_time = now.strftime("%H:%M:%S")
        print("[{}] Sending a socket/json".format(current_time))
        #print(stop_websocket_thread)

        await websocket.send(json.dumps(ObjectListJson))
        await asyncio.sleep(1)  # Send updates every second
    exit()

def runHorizontalChart_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(sendDataToHorizontalChart, "127.0.0.1", 8051)
    loop.run_until_complete(start_server)
    loop.run_forever()
    

if __name__ == "__main__":
    horizontalChart_thread = threading.Thread(target=runHorizontalChart_thread)
    horizontalChart_thread.start()
    
    #capture = cv2.VideoCapture(0)  # Replace 0 with the appropriate camera index or video file path
    blankImage = np.zeros((250,250,3), np.uint8)

    try:
        while not stop_websocket_thread:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            #print(current_time)
            print("[{}] main".format(current_time))

            cv2.imshow("window", blankImage) 

            if len(detectedObjects) > 0:
                for object in detectedObjects:
                    objectJson = {
                    "Category": object,
                    "Color": None,
                    "Id": None,
                        "LastState": {
                            "BoundingBox": [None, None, None, None],
                            "MapAcceleration": None,
                            "MapPosition": [None, None],
                            "MapSpeed": None,
                            "MapSpeedAngle": None,
                            "SensorPosition": [None, None],
                            "Timestamp": str(int(time.time())*1000),
                            "WGS84Position": [None, None]
                        }
                    }
                    ObjectListJson["Objects"].append(objectJson)
                else:
                    objectJson = None
                    ObjectListJson["Objects"].append(objectJson)
            
            k = cv2.waitKey(1) & 0xFF
            # press 'q' to exit
            if k == ord('q'):
                stop_websocket_thread = True
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                print(stop_websocket_thread)
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                #horizontalChart_thread.join()
                cv2.destroyAllWindows() 
                #exit()
                #break

            time.sleep(1)
    except KeyboardInterrupt:
        stop_websocket_thread = True
        cv2.destroyAllWindows() 
        #exit()
        #ret, frame = capture.read()
        #if not ret:
        #    break
        
        #global countCar 
        #iterateee = iterateee + 1

        #global globalFrame
        #globalFrame = frame

        #print("working")
        #now = datetime.now()
        #current_time = now.strftime("%H:%M:%S")
        #print(current_time)
        #time.sleep(1)