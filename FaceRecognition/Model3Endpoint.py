# import libraries
import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream


class Model3Endpoint:
    #globale variables
    global detector
    global embedder
    global recognizer
    global le 
    
    # load serialized face detector
    print("Loading Face Detector...")
    protoPath = "FaceRecognition/face_detection_model/deploy.prototxt"
    modelPath = "FaceRecognition/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch("FaceRecognition/openface_nn4.small2.v1.t7")

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open("FaceRecognition/output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("FaceRecognition/output/le.pickle", "rb").read())

    # loop over frames from the video file stream
    def face_reconition_model1 (frame,points):
        
        
    
        #frame = vs.read()
        # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        frame = imutils.resize(frame, width=600)
        
        
        # Check if bounding box points are valid
        #if not (0 <= points["start_point"]["x"] < w and
         #       0 <= points["start_point"]["y"] < h and
          #      0 <= points["end_point"]["x"] < w and
           #     0 <= points["end_point"]["y"] < h):
            #print("Invalid bounding box points. Skipping face recognition.")
            #return None
        # Extract the ROI (region of interest) based on bounding box points frm model1
        start1X = points["start_point"]["x"]
        start1Y = points["start_point"]["y"]
        end1X = points["end_point"]["x"]
        end1Y = points["end_point"]["y"]
        roi = frame[start1Y:end1Y, start1X:end1X]
        (h, w) = roi.shape[:2]
        #ff = cv2.rectangle(frame, (start1X, start1Y), (end1X, end1Y), (0, 255, 255), 1)
        #cv2.imwrite(f'F://final-integration//frames_for_subprocessor/ff-frame-{start1Y}.jpg',ff)
        #cv2.imwrite(f'F://final-integration//frames_for_subprocessor/roiframe-{start1Y}.jpg',roi)
        #print ("\n point of model 1",start1X,start1Y,end1X,end1Y,"\n")

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(roi, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        #print("detection ",detections)
        # loop over the detections
        person  = None
        content= {"Faces":[]}
        for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            #print("conf ",confidence)
            # filter out weak detections
            if confidence > 0.5:
                #print ("gowa el loop \n")
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX,startY,endX,endY)= box.astype("int")

                # extract the face ROI
                face = roi[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                #print ("\n point of model roi",fH,fW,"\n")

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
    
                # return text along with the associated probability
                text = "{}".format(name)
                percent ="{:.2f}%".format( proba * 100)
                if text is not None:
                    content.get ("Faces").append( {
                        "person_name": text,
                        "percent":percent
                    })
        person=content
        return person
        # update the FPS counter
        #fps.update()
    def face_reconition_model2(frame):
            # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()
        person  = None
        content= {"Faces":[]}
        # loop over the detections
        for i in range(0, detections.shape[2]):
            #print("conent f awel el for loop",content)
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                #print("prediction face reconition",preds)
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the associated probability
                #text = "{}: {:.2f}%".format(name, proba * 100)
                #print(text)
                #y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "{}".format(name)
                percent ="{:.2f}%".format( proba * 100)
                if text is not None:
                    #print(content)
                # Create JSON object
                    content.get ("Faces").append( {
                        "person_name": text,
                        "percent":percent
                    })
                    #print(text)
        
        person=content
        return person

    
