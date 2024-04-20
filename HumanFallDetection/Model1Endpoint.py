import os
import cv2
import time
import torch
import argparse
import numpy as np
import json

from HumanFallDetection.Utils import ResizePadding
#from CameraLoader import CamLoader, CamLoader_Q
from HumanFallDetection.DetectorLoader import TinyYOLOv3_onecls

from HumanFallDetection.PoseEstimateLoader import SPPE_FastPose
from HumanFallDetection.fn import draw_single

from HumanFallDetection.Track.Tracker import Detection, Tracker
from HumanFallDetection.ActionsEstLoader import TSSTG

class Model1Endpoint :
    global detect_model 
    global pose_model
    #global inp_pose
    global pose_model
    global tracker
    global action_model
    global resize_fn
    # DETECTION MODEL.
    inp_dets = 384

    detect_model= TinyYOLOv3_onecls(inp_dets, device='cuda')

    # POSE MODEL.
    inp_pose = '224x160'.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose('resnet50', inp_pose[0], inp_pose[1], device='cuda')

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)
    
    def preproc(self,image):
        """preprocess function for CameraLoader.
        """
        image = resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def kpt2bbox( self,kpt, ex=20):
        """Get bbox that hold on all of the keypoints (x,y)
        kpt: array of shape `(N, 2)`,
        ex: (int) expand bounding box,
        """
        return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                        kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))
    

    def detect(self,frame):
        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)
        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        #print ("detection00 \n")
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        #print ("detection11",detections,"\n")
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            #print("detectionnnnnnnnnnnnnnnnnnn2",detections,"\n")
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
            
            # Create Detections object.
            detections = [Detection(self.kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
            #print("detection3",detections,"\n")

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)
        #print("TRAKER",tracker.tracks)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            print("i",i)
            print ("loop",len(track.keypoints_list))
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)
            #print ("track_id",track_id)
            #print ("bbox",bbox[0],bbox[1],bbox[2],bbox[3])
            #print ("center",center)
            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                #print("henaaaaaaaaaaa\n")
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                print("action name",action_name)
                if action_name == "Sitting":
                    # Create JSON object
                    data = {
                        "action_name": action_name,
                        "bounding_box": {
                            "start_point": { 
                                "x" :bbox[0],
                                "y": bbox[1]
                            }
                            ,
                            "end_point": {
                                "x": bbox[2],
                                "y": bbox[3]
                            }
                        }
                    }
                    #print ("\n point of model 1",bbox[0], bbox[1], bbox[2], bbox[3],"\n")
                    return data
                #ff = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255, 255), 1)
                #cv2.imwrite(f'F://final-integration//frames_for_subprocessor/framemodel3-.jpg',ff)
            
                #action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                #if action_name == 'Fall Down':
                    #clr = (255, 0, 0)
                #elif action_name == 'Lying Down':
                    #clr = (255, 200, 0)
        