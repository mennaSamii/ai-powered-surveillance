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
from WeaponDetection.WeaponDetection import detect
import subprocess
from WeaponDetection.utils.datasets import LoadStreams, LoadImages,letterbox
from WeaponDetection.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from numpy import random
from models.experimental import attempt_load
from WeaponDetection.utils.datasets import LoadStreams, LoadImages
from WeaponDetection.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from WeaponDetection.utils.plots import plot_one_box
from WeaponDetection.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

m=240
n=240
model = attempt_load('F://final-integration//WeaponDetection//runs//train//best5.pt')
device = select_device('cpu')


class Model2Endpoint:
    def preproc(self,image,img_size,stride):
        """preprocess function for CameraLoader.
        """
        #print(image)
        # Padded resize
        image = letterbox(image, img_size, stride=stride)[0]

        # Convert
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        return image
    def detect_weapon( self, frames, weights, imgsz, trace=None,augment=None):
        # Initialize
        set_logging()
        device = select_device('0')
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, imgsz)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        #vid_path, vid_writer = None, None
        #if webcam:
            #   view_img = check_imshow()
            #  cudnn.benchmark = True  # set True to speed up constant image size inference
            # dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        #else:
            #   dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        #for path, img, im0s, vid_cap in dataset:
        im0s=frames
        img = torch.from_numpy(frames).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred,0.25, 0.45, classes=None, agnostic=None)
            print("predictoin 1",pred)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
                print ("predd",pred)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                #   if webcam:  # batch_size >= 1
                #      p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                # else:
                #    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                s=""
                # p = Path(p)  # to Path
                #save_path = str(save_dir / p.name)  # img.jpg
                #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(frames.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frames.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        # Print time (inference + NMS)
                        #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                    for *xyxy, conf, cls in reversed(det):
                        #if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        #print ("points",xywh,"\n") 
                        print(f'{names[int(cls)]} {conf:.2f}')    
                        print("names",names[int(c)])
                        if names[int(c)] == 'knife' :
                            print ("knife detected")
                            return s
                        elif names[int(c)] == 'gun':
                            print ("gun detected")
                            return s 
                        else:
                            print("no weapon detected :.(")
