import cv2
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer,Qt

import cv2
import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt  # Import Qt class from QtCore
from PyQt5.QtCore import QSize


import imutils
from imutils import paths
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import pickle
import time
import cv2
import os
import datetime
from datetime import date
from datetime import datetime
import pandas as pd


prevname="NIL"

roll1=0
roll2=0
roll3=0
roll4=0
roll5=0
roll6=0
roll7=0
roll8=0




roll1status="ABSENT"
roll2status="ABSENT"
roll3status="ABSENT"
roll4status="ABSENT"
roll5status="ABSENT"
roll6status="ABSENT"
roll7status="ABSENT"
roll8status="ABSENT"

runIs = True
sentcommand=0

atmode=0




class FaceDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.camera = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_detected = False
        self.photo_counter = 1  # Counter to keep track of captured photos
        self. createncodings()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Face Recognition and Attendance System')
        self.setFixedSize(670, 600)  # Set fixed size for the main window

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)  # Set fixed size for the label area
        self.label.setAlignment(Qt.AlignCenter)  # Align text to the center of the label
        font = self.label.font()
        font.setPointSize(20)  # Set font size to 20 (or any desired size)
        self.label.setFont(font)
        self.label.setText("TURN ON CAMERA")

        self.startButton = QPushButton('Start Camera', self)
        self.stopButton = QPushButton('Stop Camera', self)

        # Text box for entering the name
        self.nameTextBox = QLineEdit(self)
        self.nameTextBox.setPlaceholderText("Enter Id")

        # Snap button to capture photos
        self.snapButton = QPushButton('Snap', self)
        self.snapButton.clicked.connect(self.capturePhotos)

        # Set fixed size for the buttons
        button_size = QSize(100, 50)  # Adjust button size as needed
        self.startButton.setFixedSize(button_size)
        self.stopButton.setFixedSize(button_size)
        self.nameTextBox.setFixedSize(100, 30)
        self.snapButton.setFixedSize(100, 30)

        self.attendanceButton = QPushButton('Normal', self)
        self.attendanceButton.setFixedSize(100, 30)
        self.attendanceButton.setCheckable(True)  # Make the button checkable
        self.attendanceButton.setChecked(False)
        self.attendanceButton.toggled.connect(self.toggleAttendance)

        self.processButton = QPushButton('Process', self)
        self.processButton.setFixedSize(100, 30)
        self.processButton.clicked.connect(self.createncodings)

        



        button_layout = QHBoxLayout()  # Use QHBoxLayout for side-by-side alignment
        button_layout.addWidget(self.startButton)
        button_layout.addWidget(self.stopButton)
        button_layout.addWidget(self.nameTextBox)
        button_layout.addWidget(self.snapButton)
        button_layout.addWidget(self.processButton)
        button_layout.addWidget(self.attendanceButton)  # Add the attendance button

        main_layout = QVBoxLayout()  # Use QVBoxLayout for top alignment
        main_layout.addWidget(self.label)
        main_layout.addLayout(button_layout)  # Add the button layout to the main layout

        self.setLayout(main_layout)

        self.startButton.clicked.connect(self.startCamera)
        self.stopButton.clicked.connect(self.stopCamera)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)

    def createncodings(self):
        global data
 
        current_directory = os.getcwd()
        folder_path = os.path.join(current_directory, 'facesfolder')

        if os.path.exists(folder_path):
            imagePaths = list(paths.list_images(folder_path))
        else:
            print(f"The directory '{folder_path}' does not exist in the current working directory.")


        # initialize the list of known encodings and known names
        knownEncodings = []
        knownNames = []

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1,
                len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]
            print(" NAME OF THE IMAGE ")
            print(name)
            

            # load the input image and convert it from RGB (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb,
                model='hog')

            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)

            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and
                # encodings
                knownEncodings.append(encoding)
                knownNames.append(name)

        # dump the facial encodings + names to disk
        print("[INFO] serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open('encodings.pickle', "wb")
        f.write(pickle.dumps(data))
        f.close()


        print("[INFO] loading encodings + face detector...")
        data = pickle.loads(open('encodings.pickle', "rb").read())


    def startCamera(self):
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)

        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                height, width, channel = frame.shape
                self.label.setFixedSize(width, height)  # Set a fixed size for the label
                self.label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Set size policy for the label
                self.timer.start(10)  # 10ms interval
                self.face_detected = True
        else:
            print("Error: Camera not accessible.")

    def stopCamera(self):
        self.timer.stop()
        self.face_detected = False
        if self.camera.isOpened():
            self.camera.release()
            self.label.clear()
            self.label.setAlignment(Qt.AlignCenter)  # Align text to the center of the label
            font = self.label.font()
            font.setPointSize(20)  # Set font size to 20 (or any desired size)
            self.label.setFont(font)
            self.label.setText("TURN ON CAMERA")

    def updateFrame(self):
        global atmode
        global roll1,roll2,roll3,roll4,roll5,roll6,roll7,roll8,roll1status,roll2status,roll3status,roll4status,roll5status,roll6status,roll7status,roll8status
        if self.face_detected and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                if atmode==0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = img.shape
                    bytesPerLine = ch * w
                    qImg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qImg)
                    self.label.setPixmap(pixmap)  # Update label pixmap only when camera is started and frames are received
                elif atmode==1:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

                    # compute the facial embeddings for each face bounding box
                    encodings = face_recognition.face_encodings(rgb, boxes)
                    names = []

                     # loop over the facial embeddings
                    for encoding in encodings:
                        # attempt to match each face in the input image to our known
                        # encodings
                        matches = face_recognition.compare_faces(data["encodings"],
                            encoding)
                        name = "Unknown"


                        # check to see if we have found a match
                        if True in matches:
                            # find the indexes of all matched faces then initialize a
                            # dictionary to count the total number of times each face
                            # was matched
                            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                            counts = {}

                            # loop over the matched indexes and maintain a count for
                            # each recognized face face
                            for i in matchedIdxs:
                                name = data["names"][i]
                                counts[name] = counts.get(name, 0) + 1

                            # determine the recognized face with the largest number
                            # of votes (note: in the event of an unlikely tie Python
                            # will select first entry in the dictionary)
                            name = max(counts, key=counts.get)
                        
                        # update the list of names
                        names.append(name)

                    # loop over the recognized faces
                    for ((top, right, bottom, left), name) in zip(boxes, names):
                        # draw the predicted face name on the image
                        #cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                        print(" RECOGNIZED ", name);
                        if name=="1" and roll1==0:
                            roll1=1
                            roll1status="PRESENT"
                            print("---- ROLL 1 ATTENDANCE TAKEN ---")
                        elif name=="1" and roll1==1:
                            print("### ROLL 1 ALREADY ATTENDANCE NOTED ### ")

                        if name=="2" and roll2==0:
                            roll2=1
                            roll2status="PRESENT"
                            print("---- ROLL 2 ATTENDANCE TAKEN ---")
                        elif name=="2" and roll2==1:
                            print("### ROLL 2 ALREADY ATTENDANCE NOTED ### ")

                        if name=="3" and roll3==0:
                            roll3=1
                            roll3status="PRESENT"
                            print("---- ROLL 3 ATTENDANCE TAKEN ---")
                        elif name=="3" and roll3==1:
                            print("### ROLL 3 ALREADY ATTENDANCE NOTED ### ")

                        if name=="4" and roll4==0:
                            roll4=1
                            roll4status="PRESENT"
                            print("---- ROLL 4 ATTENDANCE TAKEN ---")
                        elif name=="4" and roll4==1:
                            print("### ROLL 4 ALREADY ATTENDANCE NOTED ### ")    

                        if name=="5" and roll5==0:
                            roll5=1
                            roll5status="PRESENT"
                            print("---- ROLL 5 ATTENDANCE TAKEN ---")
                        elif name=="5" and roll5==1:
                            print("### ROLL 5 ALREADY ATTENDANCE NOTED ### ")

                        if name=="6" and roll6==0:
                            roll6=1
                            roll6status="PRESENT"
                            print("---- ROLL 6 ATTENDANCE TAKEN ---")
                        elif name=="6" and roll6==1:
                            print("### ROLL 6 ALREADY ATTENDANCE NOTED ### ")    

                        if name=="7" and roll7==0:
                            roll7=1
                            roll7status="PRESENT"
                            print("---- ROLL 7 ATTENDANCE TAKEN ---")
                        elif name=="7" and roll7==1:
                            print("### ROLL 7 ALREADY ATTENDANCE NOTED ### ") 


                        if name=="8" and roll8==0:
                            roll8=1
                            roll8status="PRESENT"
                            print("---- ROLL 8 ATTENDANCE TAKEN ---")
                        elif name=="8" and roll8==1:
                            print("### ROLL 8 ALREADY ATTENDANCE NOTED ### ")     




                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = img.shape
                    bytesPerLine = ch * w
                    qImg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qImg)
                    self.label.setPixmap(pixmap)  # Update label pixmap only when camera is started and frames are received



    def capturePhotos(self):
        name = self.nameTextBox.text().strip()  # Get the name from the text box
        if name:
            # Create the facesfolder if it does not exist
            if not os.path.exists('facesfolder'):
                os.makedirs('facesfolder')

            # Create a folder for the person inside the facesfolder
            person_folder = os.path.join('facesfolder', name)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)

            while self.photo_counter <= 20:
                ret, frame = self.camera.read()
                if ret:
                    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y + h, x:x + w]
                        # Save the face image with the specified name and photo_counter inside the person's folder
                        image_path = os.path.join(person_folder, f'{name}_{self.photo_counter}.jpg')
                        cv2.imwrite(image_path, face_img)
                        self.photo_counter += 1

        else:
            print("Please enter a name before capturing photos.")

        self.photo_counter = 1  # Reset the photo counter after capturing 20 photos

    def toggleAttendance(self, state):
        global atmode
        global roll1,roll2,roll3,roll4,roll5,roll6,roll7,roll8,roll1status,roll2status,roll3status,roll4status,roll5status,roll6status,roll7status,roll8status
        if state:
            self.attendanceButton.setText('Attendance')  # Change button text to 'Attendance' when checked
            atmode=1
        else:
            self.attendanceButton.setText('Normal')  # Reset button text to 'Normal' when unchecked
            if atmode==1:
                today = datetime.now()
                df = pd.DataFrame({'Name': ['1', '2', '3', '4', '5', '6', '7', '8'], 'Attendance': [roll1status, roll2status, roll3status, roll4status, roll5status, roll6status, roll7status, roll8status], 'Date': [today, today, today, today, today, today, today, today]})

                # read  file content
                file_path = os.path.join(os.getcwd(), 'attendancelog.xlsx')
                reader = pd.read_excel(file_path)

                # create writer object
                # used engine='openpyxl' because append operation is not supported by xlsxwriter
                writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists="overlay")

                # append new dataframe to the excel sheet
                df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1)

                # close file
                writer.close()

                roll1=0
                roll2=0
                roll3=0
                roll4=0
                roll5=0
                roll6=0
                roll7=0
                roll8=0


                roll1status="ABSENT"
                roll2status="ABSENT"
                roll3status="ABSENT"
                roll4status="ABSENT"
                roll5status="ABSENT"
                roll6status="ABSENT"
                roll7status="ABSENT"
                roll8status="ABSENT"

                atmode=0

        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())
