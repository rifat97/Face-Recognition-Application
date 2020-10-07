import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
rec = cv2.face.LBPHFaceRecognizer_create();   #Local Binary Pattern Histogram(LBPH)
rec.read("recognizer\\trainingData.yml")
path = 'dataSet'

def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile
    

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        profile = getProfile(id)
        if(profile!=None):
            cv2.putText(img, str(profile[1]),(x,y+h+30), font, 1,(119,51,255),2,cv2.LINE_AA)
            cv2.putText(img, str(profile[2]),(x,y+h+60), font, 1,(0,255,255),2,cv2.LINE_AA)
            cv2.putText(img, str(profile[3]),(x,y+h+90), font, 1,(0,255,255),2,cv2.LINE_AA)
        else:
            cv2.putText(img, "Unknown", (x,y+h+30), font, 1,(0,0,255),3,cv2.LINE_AA)
    cv2.imshow("Face Recognizer", img)
    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()


