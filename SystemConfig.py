import os
import cv2
import time
import torch
import numpy as np
import shutil 
import sys
from queue import Queue
from FaceRecognition.CameraLoader import CamLoader,CamLoader_Q
class SystemConfig:
    def add_user (self):
        fps_time = 0
        # Specify the directory where you want to create the new folder or existing folders 
        directory = "F:/final-integration/FaceRecognition/dataset"
        ch=input("To copy from source file enter:1\n""To open camera enter:2\n")
        if ch == "1":
            
            cam2=CamLoader_Q(0).start()
            # Ask for the name of the person 
            name = input("What is the name you want to add ? ")
            path =os.path.join(directory, name)
            # Check if the directory exists or not  
            if os.path.exists(path):
                option =input ("This name already exists do you want to update images on "+name+"data ? \n").lower()        
                if option =="yes" :
                    source_dir=input("Put the path of your folder ")  
                    for filename in os.listdir(source_dir):
                        # Check if it's an image file 
                        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):  
                            # Construct full paths for source and destination files
                            source_path = os.path.join(source_dir, filename)
                            dest_path = os.path.join(path, filename)
                            print("Started copying...")
                            # Copy the image file
                            shutil.copy2(source_path, dest_path)  # Preserves file metadata (optional)
                            # Alternatively, use shutil.copy() for simpler copying
                            print(f"Copied '{filename}' from {source_dir} to {path}")        
                elif option == "no":
                    print("No problem, try again :)")
            elif not os.path.exists(path):    
                # Create a new directory with the user's name
                #new_directory = os.path.join(path, name)
                os.mkdir(path)
                print(f"A new directory named '{name}' has been created in {path}.") 
                source_dir=input("Put the path of your folder ")        
                for filename in os.listdir(source_dir):
                    # Check if it's an image file 
                    if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):  
                        # Construct full paths for source and destination files
                        source_path = os.path.join(source_dir, filename)
                        dest_path = os.path.join(path, filename)
                        print("Started copying...")
                        # Copy the image file
                        shutil.copy2(source_path, dest_path)  # Preserves file metadata (optional)
                        # Alternatively, use shutil.copy() for simpler copying
                        print(f"Copied '{filename}' from {source_dir} to {path}")
                        if cv2.waitKey(3) & 0xFF == ord('q'):
                            break
                cam2.stop()
                cv2.destroyAllWindows()
        elif ch == "2":
            # Using threading.
            cam = CamLoader(0).start()
            # Ask for the name of the person 
            name = input("What is the name? ")
            path =os.path.join(directory, name)
            # Check if the directory exists or not  
            if os.path.exists(path):
                option =input ("this name already exists do you want to update images on "+name+" data ? \n").lower()
                if option =="yes" :
                    # Update the file (e.g., write new content)
                    print("Opening camera...")   
                elif option == "no":
                    print("no problem, try again :)")
                    sys.exit(1)
            elif not os.path.exists(path):    
                # Create a new directory with the user's name
                new_directory = os.path.join(directory, name)
                os.mkdir(new_directory)
                print(f"A new directory named '{name}' has been created in {directory}.")         
                print("Opening camera...")
            while cam.grabbed():
                frames = cam.getitem()

                frames = cv2.putText(frames, 'FPS: %f' % (1.0 / (time.time() - fps_time)),
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                fps_time = time.time()
                os.chdir("F:/final-integration/FaceRecognition/dataset/"+name)
                file='frame'+ str(fps_time) +'.jpg'
                cv2.imwrite(file,frames)
                cv2.imshow('frame', frames)

                if cv2.waitKey(3) & 0xFF == ord('q'):
                    break
            cam.stop()
            cv2.destroyAllWindows()
        else: 
            print ("Try again please")
    def remove_user (self):
        #to enter the user you want to remove 
        directory = "F:/final-integration/FaceRecognition/dataset"
        name = input("What is the name you want to  remove? ")
        folder_path =os.path.join(directory, name)
        # Check if the path exists
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return

        # Check if it's a folder
        if not os.path.isdir(folder_path):

            print(f"Path '{folder_path}' is not a folder.")
            return
        try:
            shutil.rmtree(folder_path)
            print("user "+name+" removed ")
        except OSError as e:
            print(f"Error deleting folder: {e}")
if __name__ == '__main__':
    choice=input("do you want to remove or add user ").lower()
    sy=SystemConfig()
    if choice== "remove":
        sy.remove_user()
    elif choice== "add":
        sy.add_user()
    