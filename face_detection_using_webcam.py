import cv2
from random import randrange

# load some trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# to capture video from webcam
webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture("/home/hacker07/Videos/Sex.Education.S02E05.720p.NF.WEB-DL.Hin-Eng.mkv")

# iterate forever
while True:
    # read current frame
    successful_frame_read, frame = webcam.read()

    # converting frames to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(225), randrange(225), randrange(225)), 3)

    cv2.imshow("face detector", frame)
    key = cv2.waitKey(1)

    if key in [81, 113]:
        break

# release webcam
webcam.release()
