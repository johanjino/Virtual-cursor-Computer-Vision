# import necessary packages

import cv2
#import numpy as np
import mediapipe as mp


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils



# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _ , frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
    
    else:
        landmarks = [None for i in range(21)]
    
    
    #relative error calculation to predict right click of mouse
    
    if landmarks[0]!=None:
        rel_err_x = ((abs(landmarks[4][0]-landmarks[8][0]))/landmarks[4][0])*100
        rel_err_y = ((abs(landmarks[4][1]-landmarks[8][1]))/landmarks[4][1])*100
    else:
        rel_err_x=100
        rel_err_y=100
        
    # show the prediction on the frame
        
    cv2.putText(frame, str((rel_err_x<10) & (rel_err_y<10)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
