from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import requests
import base64

# Constants for the Flask API endpoint
API_ENDPOINT = "http://localhost:5000/detect_face"

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.150:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')
        frame_skip = 5  # skip every 5 frames
        counter = 0
        while True:
            ret, cv_img = cap.read()
            if ret:
                counter += 1
                if counter % frame_skip == 0:
                    self.change_pixmap_signal.emit(cv_img)


class APIWorker(QThread):
    finished_signal = pyqtSignal(int, dict,np.ndarray)

    def __init__(self, img):
        super().__init__()
        self.img = img

    def run(self):
        _, img_encoded = cv2.imencode('.jpg', self.img)
        jpg_as_text = base64.b64encode(img_encoded)
        response = requests.post(API_ENDPOINT, files={'file': jpg_as_text})
        if response.status_code == 200:
            data = response.json()
            self.finished_signal.emit(response.status_code, data,self.img)
        else:
            self.finished_signal.emit(response.status_code, {},self.img)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("API Based Face Detection Tester")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)



        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.api_workers = []



    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        api_worker = APIWorker(cv_img)
        api_worker.finished_signal.connect(self.process_api_response)
        api_worker.finished.connect(self.cleanup_api_worker)
        api_worker.start()
        self.api_workers.append(api_worker)

    @pyqtSlot()
    def cleanup_api_worker(self):
        sender = self.sender()  # This gets the APIWorker object that sent the signal.
        self.api_workers.remove(sender)

    @pyqtSlot(int, dict,np.ndarray)
    def process_api_response(self, status_code, data,frame_backup):
        if status_code == 200:
            for face in data['faces']:
                x, y, w, h = face['x'], face['y'], face['width'], face['height']
                cv2.rectangle(frame_backup, (x, y), (x + w, y + h), (0, 255, 0), 2)
            qt_img = self.convert_cv_qt(frame_backup)
            self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())