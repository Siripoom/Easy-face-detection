import cv2

img = cv2.imread('char.jfif') # get file path

face_model = cv2.CascadeClassifier('face-detect-model.xml') # call model face-detect-model

faces = face_model.detectMultiScale(img) # execute face detect

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    
cv2.resize(img,(400,400))
cv2.imshow('image',img)

cv2.waitKey(0) # any keys to close
cv2.destroyAllwindows()