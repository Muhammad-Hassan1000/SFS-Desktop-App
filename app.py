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
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)

# ------------------------------- WEBCAM INPUT ------------------------------------------ #
        # Create a webcam widget
        # webcam_widget = QWidget(self)
        # webcam_layout = QHBoxLayout(webcam_widget)
        
        # Create a label to display the webcam feed
        self.webcamLabel = QLabel(self)
        self.webcamLabel.setScaledContents(True)
        self.webcamLabel.setStyleSheet("padding-left: 30px")
        self.webcamLabel.setFixedWidth(780)
        self.webcamLabel.setFixedHeight(560)
        
        # self.setCentralWidget(self.webcamLabel)
        main_layout.addWidget(self.webcamLabel, alignment=Qt.AlignmentFlag.AlignVCenter)
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

        # Create layout for right half of screen
        right_layout = QGridLayout()
        right_layout.setSpacing(10)
        main_layout.addLayout(right_layout, 1)

        # Create label for category label
        self.category_label = QLabel("Category")
        self.category_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.category_label.setStyleSheet("font-family: Bebas Neue; font-size: 30pt; font-weight: bold")
        right_layout.addWidget(self.category_label, 0, 0, 3, 3)

        # Create label for category name
        self.category_name = QLabel()
        self.category_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.category_name.setStyleSheet("font-family: Bebas Neue; font-size: 30pt; font-weight: bold")
        right_layout.addWidget(self.category_name, 0, 1, 3, 3)

        # Create label for price label
        self.price_label = QLabel("Price")
        self.price_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.price_label.setStyleSheet("font-family: Bebas Neue; font-size: 30pt; font-weight: bold")
        right_layout.addWidget(self.price_label, 1, 0, 3, 3)

        # Create label for price value
        self.price_value = QLabel()
        self.price_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.price_value.setStyleSheet("font-family: Bebas Neue; font-size: 30pt; font-weight: bold")
        right_layout.addWidget(self.price_value, 1, 1, 3, 3)

    
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

        # Convert the frame to a QImage
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)

        # Set the image to the webcam label
        self.webcamLabel.setPixmap(QPixmap.fromImage(image))

    def predict(self, image_path):
        model = load_model('model_inceptionresnetv2.h5')
        img = image.load_img(image_path, target_size=(299,299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)[0]
        category = prediction
        return category

    def updateOutput(self, result):
        self.category_name.setText(str(api.category_dict[str(result)]))
        self.price_value.setText(str(api.price_dict[str(result)]))
    
# -----------------------VIDEO INPUT ---------------------------------------------------- #
        # Create video instance and add to layout
    #     self.video_instance = QMediainstance()
    #     self.video_widget = QVideoWidget()
    #     self.video_widget.setFixedSize(720, 480)
    #     self.video_instance.setVideoOutput(self.video_widget)
    #     main_layout.addWidget(self.video_widget, 2, alignment=Qt.AlignmentFlag.AlignVCenter)

    #     # Create layout for right half of screen
    #     right_layout = QVBoxLayout()
    #     right_layout.setSpacing(10)
    #     main_layout.addLayout(right_layout, 1)

    #     # Create label for textual data and add to right layout
    #     self.category_label = QLabel("Textual Data")
    #     self.category_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    #     right_layout.addWidget(self.category_label)

    #     # Create upload video button and add to right layout
    #     self.upload_button = QPushButton("Upload Video")
    #     self.upload_button.clicked.connect(self.upload_video)
    #     right_layout.addWidget(self.upload_button)

    # def upload_video(self):
    #     options = QFileDialog.Option(value=0x00000008)
    #     options |= QFileDialog.Option.DontUseNativeDialog
    #     file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.wmv)", options=options)
    #     if file_name:
    #         self.video_instance.setSource(QUrl.fromLocalFile(file_name))
    #         self.video_instance.setLoops(-1)
    #         self.video_instance.play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    instance = VehicleDetection()
    instance.show()
    thread = threading.Thread(target=api.thread_function)
    thread.start()
    sys.exit(app.exec())
