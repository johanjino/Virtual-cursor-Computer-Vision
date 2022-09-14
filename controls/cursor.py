
from pynput.mouse import Button,Controller
 

def mouse_task(index,landmarks):
    """
    
    Parameters
    ----------
    index : index of hand signal identified 
            by tensorflow predict
            
    landmarks : list of handlandmarks
            with its x and y coordinates
    
    Returns
    -------
    None
    void function
    
    """
    mouse = Controller()
    if index == 1:
        mouse.click(Button.left,1)
    if index == 2:
        mouse.click(Button.right,1)
    if index == 6:
        mouse.position = ((landmarks[8][0]/480)*1535 , (landmarks[8][1]/640)*863)
    


#initialize data
#data = ["None", "Left Click","Right Click","Select","Scroll Up","Scroll Down","Move"]