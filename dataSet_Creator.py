import cv2
import sqlite3
import numpy as np

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

def inserOrUpdate(Id, Name):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID="+str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if(isRecordExist == 1):
        cmd = "UPDATE People SET Name="+str(Name)+" WHERE ID="+str(Id)
    else:
        cmd = "INSERT INTO People(ID, Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

id = input("Enter user ID: ")
name = input("Enter your name: ")
inserOrUpdate(id, name)

sampleNum = 0
count = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        count += 1
        sampleNum = sampleNum + 1
        cv2.imwrite("dataSet/User." + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        cv2.putText(img, str(count), (x+w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if(sampleNum>99):
        break

cam.release()
cv2.destroyAllWindows()
print("Collecting samples completed!")


