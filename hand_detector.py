import numpy as np
import mediapipe as mp
import data_trainer


class hand:
    def __init__(self):
        self.hands = mp.solutions.hands
        self.detection = self.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.draw = mp.solutions.drawing_utils
        try:
            self.model = data_trainer.findmodel("created_model")
        except OSError:
            self.model = None
            print("No Model found")
    
    def detect(self, framergb):
        self.result = self.detection.process(framergb)
        return self.result

    def findmodel(self, filename):
        self.model = data_trainer.findmodel(filename)
        return self.model

    def create_model(self, filename):
        self.model = data_trainer.createmodel(filename)
        return self.model

    def savedata(self, id, filename):
        data_trainer.savedata(self.result.multi_hand_landmarks, id, filename)

    def normalise_data(self, array):
        return data_trainer.normalise_data(array)

    

