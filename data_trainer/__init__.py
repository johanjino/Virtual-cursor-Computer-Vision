
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.spatial import distance



class Callback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True



def normalise_data(array):
    """
    Parameters
    ----------
    array : np.array that contains pixel locations of all hand points
    
    Returns
    --------
    
    array : normalised values of all points (i.e. relative positions)
    
    """
    
    pivot = array[0]
    new_array=[]
    for handpoint in array:
        dst = distance.euclidean(handpoint, pivot)
        new_array.append(dst)
        

    new_array = np.array(new_array)
    maximum_col = np.amax(new_array)
    new_array = new_array/maximum_col
    
    
    return new_array
    
    
    

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
                new_set.append([lms.x,lms.y])
        new_set=np.array(new_set)
        
        new_set=normalise_data(new_set)

    
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


def findmodel(file):
    """
    
    Parameters
    ----------
    file : folder name where model is stored

    Returns
    -------
    model : tensorflow prevriously created nural net
    
    """
    model = keras.models.load_model(file)
    print("Saved model found!")
    return model




def createmodel(file):
    """
    
    Parameters
    ----------
    file : file name where dataset is stored

    Returns
    -------
    model : tensorflow created and trained nural net
    
    """
    
    f = open(file , 'rb')
    (training_images, training_labels) = pickle.load(f)
    f.close()
    model = keras.Sequential([keras.layers.Flatten(input_shape=[21]),
                              keras.layers.Dense(84,activation=tf.nn.relu),    
                              keras.layers.Dense(10,activation=tf.nn.softmax)])


    model.compile(optimizer = keras.optimizers.Adam(),
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    call_back = Callback()
    
    model.fit(training_images, training_labels, epochs=50, callbacks = [call_back])
    model.save("created_model")
    print("Model successfull")
    return model
