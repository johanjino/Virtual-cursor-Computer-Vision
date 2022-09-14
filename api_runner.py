import os
import multiprocessing


def launch_backend():
    print("RUNNING BACKEND..............")
    os.system("python test_api.py & electron .main.js &")

def launch_app(n):
    print("LAUNCHING APP................")
    os.system("electron ./main.js")
    

if __name__ == "__main__":

# creating multiple processes

    proc1 = multiprocessing.Process(target=launch_backend)
    #proc2 = multiprocessing.Process(target=launch_app)

    # Initiating process 1

    proc1.start()

    # Initiating process 2

    #proc2.start()

    # Waiting until proc1 finishes

    #proc1.join()

    # Waiting until proc2 finishes

    #proc2.join()