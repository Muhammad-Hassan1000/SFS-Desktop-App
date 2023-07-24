import sys
import os
import cv2
import datetime
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QGridLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog 
# from PyQt6.QtMultimedia import QMediainstance
from PyQt6.QtMultimediaWidgets import QVideoWidget
import threading
import api

# category_dict = api.category_dict
# price_dict = api.price_dict

# Load the pre-trained SSD model
model_path = 'mobilenet_iter_73000.caffemodel'
prototxt_path = 'deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


class VehicleDetection(QMainWindow):
    def __init__(self):
        super().__init__()

        global category_dict
        global price_dict

        category_dict = {'0': 'Large', '1':'Medium', '2':'Small'}
        price_dict = {'0': '300', '1': '250', '2': '200'}
        
        # Set window title and size
        self.setWindowTitle("Subsidized Fueling System - SFS")
        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create layout for central widget
        main_layout = QGridLayout(central_widget)
        main_layout.setSpacing(10)

# ------------------------------- WEBCAM INPUT ------------------------------------------ #
        # Create a webcam widget
        # webcam_widget = QWidget(self)
        # webcam_layout = QHBoxLayout(webcam_widget)
        
        # Create a label to display the webcam feed
        self.webcamLabel = QLabel(self)
        self.webcamLabel.setScaledContents(True)
        self.webcamLabel.setStyleSheet("padding-left: 30px; padding-bottom: 30px")
        self.webcamLabel.setFixedWidth(780)
        self.webcamLabel.setFixedHeight(560)
        self.webcamLabel.setContentsMargins(30, 0, 10, 30)
        # self.setCentralWidget(self.webcamLabel)
        main_layout.addWidget(self.webcamLabel, 1, 0, 3, 1, Qt.AlignmentFlag.AlignTop)
        # webcam_layout.addWidget(self.webcamLabel)
        # main_layout.addWidget(webcam_widget)

        # Set up the webcam
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 780)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 560)

        # Set up a timer to update the webcam feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateWebcam)
        self.timer.start(16)

        # Create label for rate list
        self.rate_label = QLabel("Rate List")
        self.rate_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.rate_label.setContentsMargins(630, 20, 0, 0)
        self.rate_label.setStyleSheet("font-family: Bebas Neue; font-size: 20pt; font-weight: bold")
        main_layout.addWidget(self.rate_label, 0, 0)

        # Create label for rate values
        self.rate_value = QLabel("{}: {}, \n{}: {}, \n{}: {}".format(api.category_dict["2"], api.price_dict["2"], api.category_dict["1"], api.price_dict["1"], api.category_dict["0"], api.price_dict["0"]))
        self.rate_value.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.rate_value.setContentsMargins(0, 20, 0, 0)
        self.rate_value.setStyleSheet("font-family: Bebas Neue; font-size: 20pt;")
        main_layout.addWidget(self.rate_value, 0, 1, 1, 2)

        # Create a QTimer that triggers every second to update rate list
        self.rate_timer = QTimer(self)
        self.rate_timer.setInterval(1000)
        self.rate_timer.timeout.connect(self.updateRates)

        # Start the rate timer
        self.rate_timer.start()
        
        # Create layout for right half of screen
        # right_layout = QGridLayout()
        # right_layout.setSpacing(10)
        # main_layout.addLayout(right_layout, 0, 1)

        # Create label for category label
        self.category_label = QLabel("Category")
        self.category_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.category_label.setContentsMargins(150, 100, 0, 20)
        self.category_label.setStyleSheet("font-family: Bebas Neue; font-size: 30pt; font-weight: bold")
        main_layout.addWidget(self.category_label, 1, 1, 2, 1)

        # Create label for category name
        self.category_name = QLabel()
        self.category_name.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.category_name.setContentsMargins(10, 100, 0, 20)
        self.category_name.setStyleSheet("font-family: Bebas Neue; font-size: 30pt; font-weight: bold")
        main_layout.addWidget(self.category_name, 1, 2, 2, 1)

        # Create label for price label
        self.price_label = QLabel("Price")
        self.price_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.price_label.setContentsMargins(150, 100, 0, 20)
        self.price_label.setStyleSheet("font-family: Bebas Neue; font-size: 30pt; font-weight: bold")
        main_layout.addWidget(self.price_label, 2, 1, 2, 1)

        # Create label for price value
        self.price_value = QLabel()
        self.price_value.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.price_value.setContentsMargins(10, 100, 0, 20)
        self.price_value.setStyleSheet("font-family: Bebas Neue; font-size: 30pt; font-weight: bold")
        main_layout.addWidget(self.price_value, 2, 2, 2, 1)

    
    def keyPressEvent(self, event):
        # Check if the Escape key was pressed
        if event.key() == Qt.Key.Key_Escape:
            print("Esc pressed, closing.....")
            self.close()
            api.end_api = True

        # Check if Space key was pressed
        if event.key() == Qt.Key.Key_Space:
            # Read a frame from the webcam
            ret, frame = self.webcam.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                return

            # Getting present working directory's path
            absolute_path = os.getcwd()
            
            # Checking Path to save images
            if not os.path.exists(os.path.join(absolute_path, "SFS Captures")):
                os.mkdir(os.path.join(absolute_path, "SFS Captures"))
            
            # Saving Image captured
            image_path = "SFS Captures/" + datetime.datetime.now().strftime("%d-%m-%y %H-%M-%S") + ".jpg"
            cv2.imwrite(image_path, frame)

            # Predict on the captured image
            category = self.predict(image_path)
            print("Spacebar pressed!")

            self.updateOutput(category)
            
    
    def updateWebcam(self):
        # Read a frame from the webcam
        ret, frame = self.webcam.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            return

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        # Set the input to the model and perform detection
        net.setInput(blob)
        detections = net.forward()

        # Process detections and draw bounding boxes on the frame
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.9:  # Set the confidence threshold here
                
                class_id = int(detections[0, 0, i, 1])
                class_name = 'Car'  # You can use a list of class names if needed
                color = (255, 0, 0)  # BGR color for bounding box (blue in this case)
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x1, y1, x2, y2 = box.astype('int')
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
                cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # self.predict(frame[y1:y2, x1:x2])

                # Predict on the captured image in a new thread
                thread = threading.Thread(target=self.predict, args=(frame,))
                thread.start()

        # Show the frame with bounding boxes
        #cv2.imshow('Car Detection', frame)

        # Convert the frame to a QImage
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)

        # Set the image to the webcam label
        self.webcamLabel.setPixmap(QPixmap.fromImage(image))

    def updateRates(self):
        self.rate_value.setText("{}: {}, \n{}: {}, \n{}: {}".format(api.category_dict["2"], api.price_dict["2"], api.category_dict["1"], api.price_dict["1"], api.category_dict["0"], api.price_dict["0"]))
    count_thread=0
    def predict(self, image_path):
        print(self.count_thread,"000000000000000")
        if self.count_thread>0:
            return

        self.count_thread+=1
        print(self.count_thread,"111111111")
        model = load_model('model_inceptionresnetv2.h5')
        # img = image.load_img(image_path, target_size=(299,299))
        # x = image.img_to_array(img)
        d = cv2.resize(image_path, (299, 299))
        x = np.expand_dims(d, axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)[0]
        category = prediction
        print("Category: ", category, self.count_thread)
        # return category
        self.category_name.setText(api.category_dict[str(category)])
        self.price_value.setText(api.price_dict[str(category)])
        self.count_thread=0

    def updateOutput(self, result):
        self.category_name.setText(api.category_dict[str(result)])
        self.price_value.setText(api.price_dict[str(result)])
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    instance = VehicleDetection()
    instance.show()
    thread = threading.Thread(target=api.thread_function)
    thread.start()
    sys.exit(app.exec())
