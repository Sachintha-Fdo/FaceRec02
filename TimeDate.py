import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
#path to the Initial images
path = 'primage'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

#encoding
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#record the details of detected images
def markTimeDate(name):
    with open('EnteringTime.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')
        print(myDataList)

encodeListKnown = findEncodings(images)
print('Encoding Completed')
#capture the video
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
#marking the 128 points of the deteced faces
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
#image tracking
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+8,y2-8),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markTimeDate(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(2)



"""#face detect
faceLoc = face_recognition.face_locations(imgbill)[0]
encodBill = face_recognition.face_encodings(imgbill)[0]
cv2.rectangle(imgbill,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#image encoding
faceLocTest = face_recognition.face_locations(imgtest)[0]
encodTest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodBill],encodTest)
faceDis = face_recognition.face_distance([encodBill],encodTest)
#"""