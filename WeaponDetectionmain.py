import os
import cv2
import time
import torch
from models.experimental import attempt_load
import torch.backends
import argparse
import numpy as np
import tensorflow as tf
from WeaponDetection.CameraLoader import CamLoader, CamLoader_Q
#from PIL import Image
from keras.preprocessing.image import  img_to_array
import threading 
from tensorflow import keras
# Import the class

import subprocess
from WeaponDetection.utils.datasets import LoadStreams, LoadImages
from WeaponDetection.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from threading import Thread
import shutil
#from Detection.Utils import ResizePadding
m=240
n=240

#model = torch.load('E:/Gun-Detection-master/runs/train/yolov7-custom5/weights/best.pt')
device = select_device('')
model = attempt_load('F://final-integration//WeaponDetection//runs//train//knifedetection014//weights//best.pt')
half = device.type != 'cpu'
frame_path='E:/saved_frames'
def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = cv2.resize(image,(m,n))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def detect_weapon(frame_path, weight_path, confidence, image_size):
    
        """
    Function to execute detect.py using a subprocess in a separate thread.

    Args:
        frame_path (str): Path to the frame image file.
        weight_path (str): Path to the model weights file.
        confidence (float): Confidence threshold for detection.
        image_size (int): Image size for detection.
        """
        os.chdir('E:/saved_frames')
        while True:
        # Loop through all files in the directory
            for filename in os.listdir(frame_path):
                # Check if it's an image file
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    print ('file name :',filename)
                    src=os.path.join(frame_path, filename)
                    subprocess.run([
                    "python", "F://final-integration//WeaponDetect.py",
                    "--weight", weight_path,
                    "--conf", confidence,
                    "--img-size", image_size,
                    "--source", src
                    ])
                   # subprocess.run([
                   # "python", "E:/Gun-Detection-master/detect.py",
                   # "--weight", weight_path2,
                   # "--conf", confidence,
                   # "--img-size", image_size,
                   # "--source", src
                   # ])
                    #shutil.move(src, 'E:/saved_frames/processed')
                    #os.system ("python E:/Gun-Detection-master/detect.py --weight "+ weight_path +" --conf "+confidence+" --img-size "+image_size+"  --source "+ image_path )
                    print("after detection ")
                        
        
        



if __name__ == '__main__':
    #par = argparse.ArgumentParser(description='wepon detection.')
    #par.add_argument('-C', '--camera',default="0", #default=source,  # required=True,  # default=2,
    #                    help='Source of camera or video file path.')
    #par.add_argument('--call model', default=0,
     #                   help='')
    #par.add_argument('--device', type=str, default='cuda',
     #                   help='Device to run model on cpu or cuda.')
    #par.add_argument('--weights', nargs='+', type=str, default='E:/Gun-Detection-master/runs/train/yolov7-custom5/weights/best.pt', help='model.pt path(s)')
    #par.add_argument('--source', type=str, default='E:/saved frames',
     #                   help='this is saved frames from cam streaming ')
    #par.add_argument('--conf', type=float, default=0.25, help='object confidence threshold')
    #args = par.parse_args()
    
    # Arguments you want to pass to detect.py
    arguments = [
    "--weight F://final-integration//WeaponDetection//runs//train//knifedetection014//weights//best.pt",
    "--conf 0.4",
    "--img-size 640",
    "--source E:/saved_frames"
    ]

    #script_path='python detect.py'
    device = 'cuda'
    cam_source = '0'
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,preprocess=preproc).start()
        
        #if KeyboardInterrupt:
    
    outvid = False
    
    fps_time = 0
    f = 0
    detect_thread = Thread(target=detect_weapon, args=(frame_path,'F://final-integration//WeaponDetection//runs//train//knifedetection014//weights//best.pt','0.4', '640',))  
    detect_thread.start()
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()
 
       

        # Show Frame.
        time_diff=(time.time() - fps_time)
        if time_diff ==0 :
            time_diff=1 
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / time_diff),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()            
        os.chdir('E:/saved_frames')
        # Detect wepon using our trrained model.
        #de=detect
        #de.detect(weights=arg1,source=arg4)
        # Construct the command to execute detect.py
        
        # Call detect.py using subprocess.run

        file='frame'+ str(fps_time) +'.jpg'
        #cv2.imwrite(file,frame)
        #print("frame saved\n")
        
        #subprocess.run([
         #   "python", "E:/Gun-Detection-master/detect.py",
          #  "--weight", "E:/Gun-Detection-master/runs/train/yolov7-custom5/weights/best.pt",
           # "--conf", str(0.4),
           # "--img-size", str(640),
          #  "--source", os.path.join('E:/saved_frames', 'frame1710704590.3983831.jpg')
        #])
        
        # Spawn a thread for weapon detection (ensure detect.py is optimized for images)
        # frame_path = os.path.join('E:/saved_frames', file)
        #cv2.imwrite(frame_path, frame)  # Save frame for detection
        # Spawn a thread for weapon detection (ensure detect.py is optimized for images)
        # Adjust confidence threshold
        

        #detect_weapon (frame_path, "E:/Gun-Detection-master/runs/train/yolov7-custom5/weights/best.pt", 0.4, 640)
            
            # Detect weapon using detect.py (optimized for images)
        
        #os.system ("python E:/Gun-Detection-master/detect.py --weight E:/Gun-Detection-master/runs/train/yolov7-custom5/weights/best.pt --conf 0.4 --img-size 640 --source E:/saved_frames/"+ file )
        #subprocess.run(["python"," E:/Gun-Detection-master/detect.py", "--weight", " E:/Gun-Detection-master/runs/train/yolov7-custom5/weights/best.pt"," --conf ", " 0.4", " --img-size ", " 640", " --source ", os.path.join(" E:/saved_frames/", file) ])        #if outvid:
        #writer.write(frame)
        #stride =int(model.stride.max())
        #dataset = LoadImages("E:/saved_frames/"+ file , img_size=640, stride=stride)
        #for path, img, im0s, vid_cap in dataset:
            #img = torch.from_numpy(img).to(device)
            #img = img.half() if half else img.float()  # uint8 to fp16/32
            #img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #if img.ndimension() == 3:
           #     img = img.unsqueeze(0)
          #  with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
         #       pred = model(img,augment='')[0]
                #print(pred)

        #model.eval()
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    #detect_thread.start()
    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
    # Clean up temporary frames after processing (optional)
    #for filename in os.listdir('E:/saved_frames'):
      #  os.remove(os.path.join('E:/saved_frames', filename))
#!python detect.py --weight /content/gdrive/MyDrive/MediumProject1/yolov7/runs/train/yolov7-custom5/weights/best.pt 
# --conf 0.4 --img-size 640 --source data/testSamples/n.jpg
     

    
    #if type(cam_source) is str: #and os.path.isfile(cam_source)
        # Use loader thread with Q for video file.
        #cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    #else:
        # Use normal thread loader for webcam.
        #cam_source=eval(input())
        #cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
     #                   preprocess=None)
        #threading = Thread(target=cam.start_prediction, args=())
        #threading.start()
    #else:
        #print ('cant open')

    # Load the TensorFlow model
    
    #detect.weights
   # model = keras.layers.TFSMLayer('E:/Gun-Detection-master/runs/train/yolov7-custom5/weights/best.pt/', call_endpoint='serving_default')

    #model = tf.keras.models.load_model('E:/Gun-Detection-master/runs/train\yolov7-custom5/weights/best.pt')
    #model2=keras.layers.TFSMLayer('E:\Gun-Detection-master\runs\train\yolov7-custom31/weights/best.pt/', call_endpoint='serving_default')
    #model2 = tf.keras.models.load_model('E:\Gun-Detection-master\runs\train\yolov7-custom31/weights/best.pt')  
    #path3='E:/saved frames/'
 
    #files=os.listdir(path3)
    #for i  in files:
            #img=i
            #print (img)
            #im = Image.open(path3 + img)
            #imrs = im.resize((m,n))
            #imrs=img_to_array(imrs)/255;
            #imrs=imrs.transpose(2,0,1);
            #imrs=imrs.reshape(3,m,n);

            #x=[]
            #x.append(imrs)
            #x=np.array(x)
            #predictions = model.predict(x)
            #predictions2= model2.predict(x)
            #print ('prediction result:',predictions,'\n features [knife ]',predictions2,'\n features [guns]' )
            
            #if np.any(predictions>0.999):
             #    print ('\n send alert wepon recognition' )
            #else:
             #    print ('\n no wepon detected fell safe :)')
           

           # for j in predictions:
                # if predictions.any(7,7,7):
                 #     print ('send alert wepon recognition' )
                 #else :
                  #   print ('no wepon detected fell safe :)')    
            #if any(flag >= 7 for (flag, _,_) in predictions):
                 #print ('send alert for knife recognition' )
            #elif any(flag >= 7 for (_,flag,_) in predictions):
                 #print ('send alert for long gun recognition' )
            #elif any(flag >= 7+ for (_, _,flag) in predictions):
                 #print ('send alert for short gun recognition' )
            #else :
                 #print ('no wepon detected fell safe :)')
            #if predictions.all(1):
            #print('send alert for wepon recognition' )
            #print ('hi',model.summary())
