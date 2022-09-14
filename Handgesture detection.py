
import cv2
import keyboard
import numpy as np
import mediapipe as mp
from time import time

import data_trainer
from controls import cursor


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


#initialize data
data = ["None", "Left Click","Right Click","Select","Scroll Up","Scroll Down","Move"]


start = False

# Initialize the webcam
cap = cv2.VideoCapture(0)


#load model if previously saved
try:
    model = data_trainer.findmodel("created_model")
except OSError:
    print("No Model found")

prev_frame = time()

while True:
    # Read each frame from the webcam
    _ , frame = cap.read()

    x, y, c = frame.shape
    
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        current_set=[]
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                if handslms == result.multi_hand_landmarks[0]:
                    pivot = [lm.x,lm.y]

                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                
                landmarks.append([lmx, lmy])
                current_set.append([lm.x,lm.y])
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        current_set=data_trainer.normalise_data(current_set)
    
        try:  
            prediction = np.array(model.predict([list(current_set)]))
            conf = np.amax(prediction)
            if conf > 0.6:    
                predict_result = np.argmax(prediction)
            else:
                predict_result = 0
    
        except NameError:
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

    # calculate fps
    newframe_time = time()
    fps = round((1/(newframe_time-prev_frame)), 2)
    prev_frame = time()

    cv2.putText(frame, "fps: {}".format(fps), (460, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 2, cv2.LINE_AA)
    
    # Show the final output
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
        data_trainer.savedata(result.multi_hand_landmarks,1,"handgesture_dataset.bin")
    elif keyboard.is_pressed('2'):
        data_trainer.savedata(result.multi_hand_landmarks,2,"handgesture_dataset.bin")
    elif keyboard.is_pressed('3'):
        data_trainer.savedata(result.multi_hand_landmarks,3,"handgesture_dataset.bin")
    elif keyboard.is_pressed('4'):
        data_trainer.savedata(result.multi_hand_landmarks,4,"handgesture_dataset.bin")
    elif keyboard.is_pressed('5'):
        data_trainer.savedata(result.multi_hand_landmarks,5,"handgesture_dataset.bin")
    elif keyboard.is_pressed('6'):
        data_trainer.savedata(result.multi_hand_landmarks,6,"handgesture_dataset.bin")
    elif keyboard.is_pressed('p'):
        model = data_trainer.findmodel("created_model")
    elif keyboard.is_pressed('s') and not start:
        start = True
        print("Starting.........")
    
    
# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()