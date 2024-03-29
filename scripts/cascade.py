import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# TODO: read this from command line

img = cv2.imread('good/530648f8798415ca680013cd_mobile.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.2, 5)

print faces

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

