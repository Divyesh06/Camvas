#Quick testing version

import cv2
import frame_processor

def main():

    cap = cv2.VideoCapture(0)
    frame_width = 1920  
    frame_height = 1080  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    first_time = True
   
    while cap.isOpened():
       
        ret, frame = cap.read()

        if first_time:
            frame_processor.init_state(frame.shape)
            first_time = False

        frame = frame_processor.process_frame(frame, True)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()