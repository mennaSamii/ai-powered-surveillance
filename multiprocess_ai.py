import multiprocessing as mp
import queue
import threading 
import os
import cv2
import time
import torch
import numpy as np
from IncidentManagement import IncidentManagement,incident_queue
from queue import Queue
from threading import Thread, Lock
from HumanFallDetection.Model1Endpoint import Model1Endpoint
from WeaponDetection.Model2Endpoint import Model2Endpoint
from FaceRecognition.Model3Endpoint import Model3Endpoint
#x=input("enter your camera id")
x=0
source=int(x)
#print (type (source))
frame_queue = queue.Queue()  # Create a queue to store frames
#src='F:/final-integration/frames/incident_fainting/frame-1713393883.932.jpg'
#ff=None
weight_path='F://final-integration//WeaponDetection//runs//train//best5.pt'
db_config = {
'host': "127.0.0.1",
'user': 'root',
'password': '_Admine1234',
'database': 'SecuritySystem'
}
#filename='F://final-integration//frames_for_subprocessor'
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
        self.frame_sb = None
        #assert for testing 
        assert self.stream.isOpened(), 'Cannot read camera source!'
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        #print("frame per seconed ",self.fps)
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
            #print("gowa el loop ")
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
        #print ("el queue",frame_queue.qsize())
        #return frame_sb
        #model1=ModelEndpoint()
        #print (frame)
        #model1.detect(frames)
            
        #print("frame_queue")


            
def frame_processor(self,fps_time):
        #print ("started frame_processor")
        while True:
            fps = time.time()
            fps_time = time.ctime()
            fps_time=fps_time[11:19]
            print ("frame time ",fps_time[11:19])
            # Optional: Convert the timestamp to a human-readable format (less precise)
            # This uses time.ctime() which provides a string representation but 
            # may not be in the desired format for all use cases.
            # readable_time = time.ctime(current_time)

            # Print the current time as the number of seconds since the epoch
            print(fps_time)

            
            
            frame=frame_queue.get()
            incidents,incident1,incident2=None,None,None
            # Wait for the next frame
            #print ("el queue",frame_queue.qsize())
            #cv2.imwrite(f'F://final-integration//frames_for_subprocessor//frame-{fps_time:.3f}.jpg', frame)
            image = frame.copy()
            image1= frame.copy()
            # Add FPS overlay (similar to your implementation)
            model1=Model1Endpoint()
            image=model1.preproc(image)
            result_queue_model1=model1.detect(image)
            print("\n result of model1",result_queue_model1,"\n")
            
            
            #detection of model2
         
            model2 =Model2Endpoint()
            image1=model2.preproc(image1,640,32)
            result_queue_model2=model2.detect_weapon(image1,weight_path,640)
            print("\n result of model2",result_queue_model2,"\n")
            
            #face recognition model3 
            if result_queue_model1 is not None  :
                model31=Model3Endpoint
                
                points = result_queue_model1["bounding_box"]
                result_queue_model3 =model31.face_reconition_model1(image,points)
                print("incident result 1 ",result_queue_model3,result_queue_model1)
                filename =(f'F://final-integration//frames//incident_fainting//frame-{fps}.jpg')
                cv2.imwrite(filename, frame)
                incident1={
                    "type":'falling down',
                    "json_model" : result_queue_model1,
                    "frame_path" : filename,
                    "frame_time" :fps_time,
                    "json_model_FaceRecognition" : result_queue_model3,
                    "Camera_ID":'0'
                    
                }
            if result_queue_model2 is not None  :
                model32=Model3Endpoint
                result_queue_model3 =model32.face_reconition_model2(frame)
                print("incedent result 2 ",result_queue_model3,result_queue_model2)
                filename=(f'F://final-integration//frames//incedent_weapon//frame-{fps}.jpg')
                cv2.imwrite(filename, frame)
                incident2={
                    "type": 'weapon detected',
                    "json_model" : result_queue_model2,
                    "frame_path" : filename,
                    "frame_time" : fps_time,
                    "json_model_FaceRecognition" : result_queue_model3,
                    "Camera_ID":'0'
                    
                }
            if incident2 is not None and incident1 is not None:
                #print("ady el incidents ",incident1,incident2)
                incidents={"incidents":[
                        incident1,
                        incident2
                ]}
            elif incident2 is not None :
                incidents={"incidents":[
                    incident2
                ]}
            elif incident1 is not None :
                incidents={"incidents":[
                    incident1
                ]}
            #incidents={'incidents': [{'type': 'falling down', 'json_model': {'action_name': 'Sitting', 'bounding_box': {'start_point': {'x': 55, 'y': 115}, 'end_point': {'x': 361, 'y': 364}}}, 'frame_path': 'F://final-integration//frames//incident_fainting//frame-1715809190.1674523.jpg', 'frame_number': 1715809190.1674523, 'json_model_FaceRecognition': {'Faces': [{'person_name': 'menna', 'percent': '83.23%'}]}, 'Camera_ID': '0'}
                                   # ,{'type': 'weapon detected', 'json_model': '1 knife, ', 'frame_path': 'F://final-integration//frames//incedent_weapon//frame-1715809190.1674523.jpg', 'frame_number': 1715809190.1674523, 'json_model_FaceRecognition': {'Faces': [{'person_name': 'menna', 'percent': '84.85%'}]}, 'Camera_ID': '0'}]}

            if incidents is not None :
                
                incident_queue.put(incidents)
            frame_queue.task_done()
            
            #if result_queue_model1 is not None:
              # cv2.imwrite(f'F://final-integration//frames//incident_fainting//frame-{fps_time:.3f}.jpg', frame)
            #if result_queue_model2 is not None:
             #   cv2.imwrite(f'F://final-integration//frames//incident_weapon//frame-{fps_time:.3f}.jpg', frame)
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break

def process_incident(self,fps_time):
    fps_time=0
    while True:
        IncidentManagement(db_config).ProcessIncident()
    

class system_manager():
    def __init__(self):
        #print("thread opened ")
        # Queues
        #self.frame_queue = queue.Queue
        #self.result_queue_model1 = queue.Queue()  # Specific queue for model 1 results
        #self.result_queue_model2 = queue.Queue()  # Specific queue for model 2 results
        #self.result_queue_model3 = queue.Queue()  # Specific queue for model 3 results
        # ... (queues for other models)

        # Threads
        #self.camera_reader = threading.Thread(target=camera_reader, args=(source))
        #self.camera_reader.daemon = True
        #self.camera_reader.start()

        
        self.frame_processor = threading.Thread(target=frame_processor, args=( self,fps_time))
        self.frame_processor.daemon = True
        self.frame_processor.start()
        self.process_incident = threading.Thread(target=process_incident, args=(self,fps_time))
        self.process_incident.daemon = True
        self.process_incident.start()


if __name__ == '__main__':
    fps_time = 0
    #print(torch.version.cuda)

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


