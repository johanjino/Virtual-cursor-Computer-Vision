import cv2
import keyboard
import numpy as np
from time import time

from hand_detector import *
from controls import cursor

#initialize data
data = ["None", "Left Click","Right Click","Select","Scroll Up","Scroll Down","Move"]


start = False


#initialize hand_detector
print("Initializing hand_detector.....")
hand_detector = hand()


# Initialize the webcam
print("Starting Camera..")
cap = cv2.VideoCapture(0)

prev_frame = time()

while True:

    # Read each frame from the webcam
    _ , frame = cap.read()

    x, y, c = frame.shape
    
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hand_detector.detect(framergb)

    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        current_set=[]
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            count = 0 
            for lm in handslms.landmark:

                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                
                landmarks.append([lmx, lmy])
                current_set.append([lm.x,lm.y])
            # Drawing landmarks on frames
            hand_detector.draw.draw_landmarks(frame, handslms, hand_detector.hands.HAND_CONNECTIONS)
        current_set=hand_detector.normalise_data(current_set)
    
        try:  
            prediction = np.array(hand_detector.model.predict([list(current_set)]))
            conf = np.amax(prediction)
            if conf > 0.8:    
                predict_result = np.argmax(prediction)
            else:
                predict_result = 0
    
        except AttributeError:
            predict_result = 0
            conf = 0
    else:
        predict_result = 0
        conf = 0
    # show the prediction on the frame
        
    cv2.putText(frame, data[predict_result], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(conf), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 2, cv2.LINE_AA)

    

    # show frame
    cv2.imshow("Output", frame) 
    cv2.waitKey(10)
    
    if start:
        try:
            cursor.mouse_task(predict_result, landmarks)
        except NameError:
            pass
    
    if keyboard.is_pressed('q'):
        start = False
        break
    elif keyboard.is_pressed('1'):
        hand_detector.savedata(1,"handgesture_dataset.bin")
    elif keyboard.is_pressed('2'):
        hand_detector.savedata(2,"handgesture_dataset.bin")
    elif keyboard.is_pressed('3'):
        hand_detector.savedata(3,"handgesture_dataset.bin")
    elif keyboard.is_pressed('4'):
        hand_detector.savedata(4,"handgesture_dataset.bin")
    elif keyboard.is_pressed('5'):
        hand_detector.savedata(5,"handgesture_dataset.bin")
    elif keyboard.is_pressed('6'):
        hand_detector.savedata(6,"handgesture_dataset.bin")
    elif keyboard.is_pressed('p'):
        model = hand_detector.create_model("handgesture_dataset.bin")
    elif keyboard.is_pressed('f'):
        model = hand_detector.findmodel("created_model")
    elif keyboard.is_pressed('s') and not start:
        start = True
        print("Starting.........")
    


# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()