from flask import Flask, render_template, Response
import cv2

import numpy as np
import mediapipe as mp
import data_trainer

app = Flask(__name__)

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


camera = cv2.VideoCapture(0)  # use 0 for web camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        
        # Read each frame from the webcam
        success , frame = camera.read()

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
                    if handslms == result.multi_hand_landmarks[0]:
                        pivot = [lm.x,lm.y]
                    # print(id, lm)
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
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    #Video streaming home page.
    return render_template('backend.html')



if __name__ == '__main__':
    app.run("localhost",1000)
