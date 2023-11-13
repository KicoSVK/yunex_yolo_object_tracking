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
import keyboard
import subprocess

def runVideoStream(inputFile, port):
    # Define the VLC command as a list of strings
    vlc_command = [
        "C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe",
        inputFile,
        "--sout",
         f"#transcode{{vcodec=h264,vb=800,acodec=mpga,ab=128,channels=2,samplerate=44100,scodec=none}}:http{{mux=ts,dst=:{port}/}}",
        "--no-sout-all",
        "--sout-keep"
    ]

    # Use subprocess to run the command
    try:
        print(f"[APP] Stream is online: http://127.0.0.1:{port}")
        subprocess.run(vlc_command, check=True)
        #print("VLC command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error executing VLC command:", e)


    logging.info("Stream with port %s finishing", port)

def kill_vlc_processes():
    try:
        subprocess.run(["taskkill", "/f", "/im", "vlc.exe"], check=True)
        print("VLC processes killed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to kill VLC processes.")

if __name__ == "__main__":

    t1 = threading.Thread(target=runVideoStream, args=("F:\\UE5, simulations\\crossroad_type_x\\scenario03\\cam01.mp4", "8080"))
    t2 = threading.Thread(target=runVideoStream, args=("F:\\UE5, simulations\\crossroad_type_x\\scenario03\\cam02.mp4", "8081"))
    #t3 = threading.Thread(target=runVideoStream, args=("F:\\UE5, simulations\\crossroad_type_x\\scenario01\\cam03.mp4", "8082"))

    t1.start()
    t2.start()
    #t3.start()
    print("[APP] All streams successfully started!")
    print("[APP] For stop all streams, PRESS KEY 'Q'")

    while True:
        try:
            if keyboard.read_event().name == 'q':
                #t1.terminate()
                #t2.terminate()
                #time.sleep(5)
                #t1.start()
                #t2.start()
                kill_vlc_processes()
                print("[APP] All streams successfully killed!")
                sys.exit()
        except KeyboardInterrupt:
            print("Exiting the program.")
            break






