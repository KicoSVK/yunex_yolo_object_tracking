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
        await asyncio.sleep(5) 
    exit()

def runVideoStream_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(sendDataToVideoStreamViewer, "127.0.0.1", 8050)
    loop.run_until_complete(start_server)
    loop.run_forever()
    

if __name__ == "__main__":
    videoStream_thread = threading.Thread(target=runVideoStream_thread)
    videoStream_thread.start()
    
    #capture = cv2.VideoCapture(0)  # Replace 0 with the appropriate camera index or video file path
    blankImage = np.zeros((250,250,3), np.uint8)

    try:
        while not stop_websocket_thread:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            #print(current_time)
            print("[{}] main".format(current_time))

            globalFrame = blankImage
            cv2.imshow("window", blankImage) 
            
            k = cv2.waitKey(1) & 0xFF
            # press 'q' to exit
            if k == ord('q'):
                stop_websocket_thread = True
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                print(stop_websocket_thread)
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                #videoStream_thread.join()
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