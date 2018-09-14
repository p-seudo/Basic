import cv2
#intialize camera
cam = cv2.VideoCapture(0)
#use the 'haarcascade_frontalface_default.xml' file to initialize the classifier
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#input user id while creating the dataset
Id=raw_input('enter your id')
sampleNum=0
while(True):
    ret, img = cam.read()
    #convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detect all faces in the video
    faces = detector.detectMultiScale(gray, 1.3, 5)
    #iterate over all faces
    for (x,y,w,h) in faces:
	#draw rectangle around all faces
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        
        sampleNum=sampleNum+1
        #save image inside dataSet directory with name User concatenated with sample number incremented above
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
	#display normal image to user
        cv2.imshow('frame',img)
    #exit when 'q' key is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    #exit when 20 image samples are stored 
    #you can edit the number if you want to use a larger dataset
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
