import cv2
from random import randrange

# load some trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# image to detect face
img = cv2.imread("external-content.duckduckgo.com2.jpeg")

# must convert the img to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangle around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(225), randrange(255), randrange(225)), 3)

# Display the image with the faces
cv2.imshow('face detector', img) 
cv2.waitKey()

