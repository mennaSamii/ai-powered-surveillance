import multiprocessing as mp
import queue
import threading
#cam loder imports 
import os
import cv2
import time
import torch
import numpy as np

from queue import Queue
from threading import Thread, Lock
from HumanDetection.ModelEndpoint import ModelEndpoint
x=input("enter your camera id")
source=int(x)
#print (type (source))
frame_queue = queue.Queue()  # Create a queue to store frames
# Define functions for camera_reader and frame_processor (replace with your implementation)
class CamLoader:
    """Use threading to capture a frame from camera for faster frame load.
    Recommend for camera or webcam.
    Args:
    camera: (int, str) Source of camera or video.,
    preprocess: (Callable function) to process the frame before return.
    """
    def __init__(self, camera, preprocess=None, ori_return=False):
        self.stream = cv2.VideoCapture(camera)
        #assert for testing 
        assert self.stream.isOpened(), 'Cannot read camera source!'
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.stopped = False 
        self.ret = False
        self.frame = None
        self.ori_frame = None
        self.read_lock = Lock()
        self.ori = ori_return

        self.preprocess_fn = preprocess

    def start(self):
        self.t = Thread(target=self.update, args=())  # , daemon=True)
        self.t.daemon = True
        self.t.start()
        c = 0
        while not self.ret:
            print("gowa el loop ")
            time.sleep(0.1)
            c += 1
            if c > 20:
                self.stop()
                raise TimeoutError('Can not get a frame from camera!!!')
        return self

    def update(self):
        
        while not self.stopped:
            ret, frame = self.stream.read()
            self.read_lock.acquire()
            #print(frame)
            self.ori_frame = frame.copy()
            if ret and self.preprocess_fn is not None:
                frame = self.preprocess_fn(frame)

            self.ret, self.frame = ret, frame
            self.read_lock.release()
            #print ("henna ya menna")
            self.camera_reader()
            

    def grabbed(self):
        """Return `True` if can read a frame."""
        return self.ret

    def getitem(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        ori_frame = self.ori_frame.copy()
        self.read_lock.release()
        
        if self.ori:
            return frame, ori_frame
        else:
            return frame
        
    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.t.is_alive():
            self.t.join()
        self.stream.release()

    def __del__(self):
        if self.stream.isOpened():
            self.stream.release()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream.isOpened():
            self.stream.release()
    
    def camera_reader(self):
        # Capture frames from camera and put them in frame_queue

            frames = self.getitem()
            
            frame_queue.put(frames)            
            #model1=ModelEndpoint()
            #print (frame)
            #model1.detect(frames)
             
            #print("frame_queue")


            
def frame_processor(fps_time, result_queue_model1, result_queue_model2):
        print ("started frame_processor")
        while True:
            fps_time = time.time()
            frame=frame_queue.get()# Wait for the next frame
            cv2.imwrite(f'F://final-integration//frames//frame-{fps_time:.3f}.jpg', frame)
            image = frame.copy()
            # Add FPS overlay (similar to your implementation)
            model1=ModelEndpoint()
            #print (frame)
            #print (type(frame ))
            
            image=model1.preproc(image)
            
            result_queue_model1=model1.detect(image)
            print("result",result_queue_model1)
            #print (detect_result)
            # Delete the saved frame (if saved earlier)
            os.remove(f'F://final-integration//frames//frame-{fps_time:.3f}.jpg')
            #cv2.imshow('frame', frame)
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break

            # Separate frame (if necessary)
            # Process frame with model1 and put result in result_queue_model1
            # Process frame with model2 and put result in result_queue_model2
            # ... (for other models)


class system_manager():
    def __init__(self):
        print("thread opened ")
        # Queues
        #self.frame_queue = queue.Queue
        self.result_queue_model1 = queue.Queue()  # Specific queue for model 1 results
        self.result_queue_model2 = queue.Queue()  # Specific queue for model 2 results
        # ... (queues for other models)

        # Threads
        #self.camera_reader = threading.Thread(target=camera_reader, args=(source))
        #self.camera_reader.daemon = True
        #self.camera_reader.start()

        
        self.frame_processor = threading.Thread(target=frame_processor, args=( fps_time,self.result_queue_model1, self.result_queue_model2))
        self.frame_processor.daemon = True
        self.frame_processor.start()


if __name__ == '__main__':
    fps_time = 0
    #modle1.detection()
    sys= system_manager()
    cam = CamLoader(source).start()
    while cam.grabbed():
        
        frames = cam.getitem()
        time_diff=(time.time() - fps_time)
        if time_diff ==0 :
            time_diff=1    
        frames = cv2.putText(frames, 'FPS: %f' % (1.0 /time_diff ),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        fps_time = time.time()
        
        
        #frame_queue.put(frames) 
        cv2.imshow('frame', frames)
        #cam.camera_reader()
        #print("frame_queue saved")
        #sys.frame_processor
        
        
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    cam.stop()
    cv2.destroyAllWindows()


