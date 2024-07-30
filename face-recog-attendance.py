import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
 
video_capture = cv2.VideoCapture(0)
 
bhamrah_image = face_recognition.load_image_file("photos/bhamrah.jpeg")
bhamrah_encoding = face_recognition.face_encodings(bhamrah_image)[0]
 
bhat_image = face_recognition.load_image_file("photos/bhat.jpeg")
bhat_encoding = face_recognition.face_encodings(bhat_image)[0]
 
chavan_image = face_recognition.load_image_file("photos/chavan.jpeg")
chavan_encoding = face_recognition.face_encodings(chavan_image)[0]
 
somani_image = face_recognition.load_image_file("photos/somani.jpeg")
somani_encoding = face_recognition.face_encodings(somani_image)[0]
 
ukey_image = face_recognition.load_image_file("photos/ukey.jpeg")
ukey_encoding = face_recognition.face_encodings(ukey_image)[0]

ranjan_image = face_recognition.load_image_file("photos/ranjanbala.JPG")
ranjan_encoding = face_recognition.face_encodings(ranjan_image)[0]

known_face_encoding = [
bhamrah_encoding,
bhat_encoding,
chavan_encoding,
somani_encoding,
ukey_encoding,
ranjan_encoding
]
 
known_faces_names = [
"Sharan Bhamrah 04",
"Jhanvi Bhat 05",
"Vedika Chavan 10",
"Ishan Somani 59",
"Krishang Ukey 63",
"HOPE YOU LIKED THE PROJECT!"
]
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()
