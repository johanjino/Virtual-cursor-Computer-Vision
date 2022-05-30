# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils



def savedata(data,label,file):
    """
    
    Parameters
    ----------
    data : multi_hand_landmarks
        mediapipe class nested list of locations
    file : file name to store data in
        binary file

    Returns
    -------
    None
    void function

    """
    new_set=[]
    if data==None:
        print("Hand not detected")
    else:
        for handslms in data:
            for lms in handslms.landmark:
                new_set.append([lms.x,lms.y,lms.z])
        new_set=np.array(new_set)


    
        try:
            f=open(file,'rb')
            training_set,training_labels=pickle.load(f)
            training_set=np.append(training_set,[new_set],axis=0)
            training_labels=np.append(training_labels,label)
        except FileNotFoundError:
            f=open(file,'wb')
            training_set=np.array([new_set])
            training_labels=np.array([label])
        except EOFError:
            training_set=np.array(np.array([new_set]))
            training_labels=np.array([label])
        f.close()
    
        f=open(file,'wb')
        pickle.dump((training_set,training_labels), f)
        f.flush()
        f.close()
        print("saved ",label)
    

def createmodel(file):
    f = open(file , 'rb')
    (training_images, training_labels) = pickle.load(f)
    f.close()
    model = tf.keras.Sequential([keras.layers.Flatten(input_shape=[21,3]),
                                 keras.layers.Dense(128,activation=tf.nn.relu),
                                 keras.layers.Dense(10,activation=tf.nn.softmax)])

    model.compile(optimizer = tf.keras.optimizers.Adam(),
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(training_images, training_labels, epochs=50)
    print("Model successfull")
    return model

data = ["None", "Right Click","Left Click","Select",]
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
        current_set=[]
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])
                current_set.append([lm.x,lm.y,lm.z])
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        #current_set=np.array(current_set)
    
        try:  
            predict_result = np.ndarray.argmax(np.array(model.predict([current_set])))
    
        except NameError:
            predict_result = 0
    else:
        predict_result = 0
    # show the prediction on the frame
        
    cv2.putText(frame, data[predict_result], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
    
    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('1'):
        savedata(result.multi_hand_landmarks,1,"handgesture_dataset.bin")
    elif cv2.waitKey(1) == ord('2'):
        savedata(result.multi_hand_landmarks,2,"handgesture_dataset.bin")
    elif cv2.waitKey(1) == ord('3'):
        savedata(result.multi_hand_landmarks,3,"handgesture_dataset.bin")
    elif cv2.waitKey(1) == ord('p'):
        model = createmodel("handgesture_dataset.bin")
        
# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()