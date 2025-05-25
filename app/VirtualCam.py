import sys
import os
import importlib

stop_signal = True
cam = None

cv2 = None
frame_processor = None

from softcam_python import softcam

def stop_camera():
    global stop_signal
    stop_signal = True

def load_modules():
    global cv2, frame_processor
    cv2 = importlib.import_module('cv2')
    frame_processor = importlib.import_module('frame_processor')

def main(started_callback):
    global stop_signal, loaded
    global cam

    if cam:
        import time
        time.sleep(1) #Wait for previous camera to disconnect
    stop_signal = False
    cam = softcam.camera(1920, 1080, 60)
    while not cam.wait_for_connection(timeout=1):
        pass
    
    if not cv2 and not frame_processor:
        load_modules()

    cap = cv2.VideoCapture(0)
    frame_width = 1920  
    frame_height = 1080  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    first_time = True
    
    
    started_callback()

    while cap.isOpened() and not stop_signal:
       
        ret, frame = cap.read()

        if first_time:
            frame_processor.init_state(frame.shape)
            first_time = False

        frame = frame_processor.process_frame(frame, False)
        
        if frame.shape != (frame_height, frame_width, 3):
           
            frame = cv2.resize(frame, (frame_width, frame_height), frame)
        
        cam.send_frame(frame)

    cam.delete()
    
    stop_signal = False

   

if __name__ == "__main__":
    main(lambda: None)