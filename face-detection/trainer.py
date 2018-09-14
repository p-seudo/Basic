#import all necessary modules
import cv2,os
import numpy as np
from PIL import Image
#initialize LBPH (Local Binary Patterns Histograms ) Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    #get all images in the given path 
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    faceSamples=[]
    
    Ids=[]
    
    for imagePath in imagePaths:
        #convert all images to grey scale
        pilImage=Image.open(imagePath).convert('L')
        #convert them into numpy array
        imageNp=np.array(pilImage,'uint8')
        #get userid from image name
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        #get faces
        faces=detector.detectMultiScale(imageNp)
        #fetch all image lables and store
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('dataSet')
#train and save data
recognizer.train(faces, np.array(Ids))
recognizer.save('trainer/trainer.yml')
