import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
#read data from trainer.yml from trainer directory
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
#use the cascade classifier
faceCascade = cv2.CascadeClassifier(cascadePath);
#intitalize camera
cam = cv2.VideoCapture(0)
#set fint styling
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
while True:
    #read and convert image to greyscale
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #detect faces
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        #draw rectangles around faces
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        #predict using the traner.yml data
        #predict method returns two values the label and match percentage
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
	#if match is greater than 50 set names according to ids        
	if(conf>50):
            if(Id==1):
                Id="person 1"
            elif(Id == 2):
                Id="person 2"
	    elif(Id == 3):
		Id="person 3"
	    elif(Id == 4 ):
		Id = "person 4"
	    else:
		Id="Unknown"
	#write name of each face
	cv2.putText(im, str(Id), (x,y+h), fontface, fontscale, fontcolor) 
    #show image
    cv2.imshow('im',im) 
    #break on 'q'
    if cv2.waitKey(10) and 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows
