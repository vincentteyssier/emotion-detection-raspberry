
import numpy as np
import cv2


faceDetect = cv2.CascadeClassifier('./retrained_data/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
ret, img = cam.read()

sampleNum = 0
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,
                                          scaleFactor=1.3,
                                          minNeighbors=5
                                          )

    for (x,y,w,h) in faces:
        sampleNum = sampleNum+1
        cv2.imwrite("collected_dataset/"+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100);
        
    cv2.imshow('face',img);
    cv2.waitKey(1);
    if sampleNum > 100:
        break
cam.release()
cv2.destroyAllWindows()