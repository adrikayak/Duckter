from __future__ import division
import numpy as np
import cv2
import cv2.cv as cv
import video
import RPi.GPIO as GPIO
import time
from datetime import datetime
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1920,1080)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(1920,1080))
dist_capture = PiRGBArray(camera, size=(1920,1080))
GPIO.setmode(GPIO.BCM)

GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_UP)


class App:


    def __init__(self, doRecord=True, showWindows=True):
        self.doRecord = doRecord
        self.show = showWindows
        self.frame = None
        self.frame_rate = camera.framerate
        self.isRecording = False

        self.trigger_time = None
        #cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv.CV_WINDOW_FULLSCREEN)
        # cv2.resizeWindow('Image', 320, 240)
            # cv2.createTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)

    def run(self):
        dist = 30 # meters
        speed = 100 # meters/second
        print "started"
        while True:
            input_state = GPIO.input(21)
            #camera.start_preview()
            if input_state == False:
                timestamp1 = time.time()
                print timestamp1
                camera.capture(dist_capture, format="bgr")
                frame_dist = dist_capture.array[180:900, 320:1600]
                dist = self.distance_to_camera(frame_dist)
                dist_capture.truncate(0)
                timestamp2 = time.time()
                print timestamp2
                time_interval = dist / speed - (timestamp2-timestamp1)
                time.sleep(time_interval)
                # print time.time()
                print('Button Pressed')
                camera.capture(rawCapture, format="bgr")
                self.frame = rawCapture.array[180:900, 320:1600]
                # currentframe = rawCapture.array[180:900, 320:1600]
                # cv2.imshow("Image", self.frame)
                self.processImage(self.frame)
                rawCapture.truncate(0)
                #for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                #    # grab the raw NumPy array representing the image, then initialize the timestamp
                #    # and occupied/unoccupied text
                #    self.frame = frame.array
                #    currentframe = frame.array
                #    self.processImage(currentframe)
            c = cv2.waitKey(1) % 0x100
            if c==27 or c == 10: #Break if user enters 'Esc'.
                break


    def processImage(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.medianBlur(gray_frame, 5)
        cimg = frame.copy() # numpy function
        height, width, depth = cimg.shape
        center_x = int(round(width/2))
        center_y = int(round(height/2))
        # print(center_x)
        # print(center_y)
        cv2.circle(cimg,(center_x,center_y),10,(0,255,0),2)

        circles = cv2.HoughCircles(gray_frame, cv.CV_HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 75)
       
        hit = 0
        if  circles is not None:
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                if (i[0] - center_x)*(i[0] - center_x) + \
                    (i[1] - center_y)*(i[1] - center_y) < \
                    i[2]*i[2]:
                    cv2.line(cimg,(width-10,10),(width-32, 50)
                        ,(0,255,0),5)
                    cv2.line(cimg,(width-50, 30),(width-32, 50)
                        ,(0,255,0),5)
                    hit += 1
                    print('You have a hit')
                # draw the center of the circle
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        if hit < 1:
            cv2.line(cimg,(width-10,10),(width-50, 50)
                ,(0,0,255),5)
            cv2.line(cimg,(width-50, 10),(width-10, 50)
                ,(0,0,255),5)
            print('You missed')
        cv2.imshow("Image", cimg)
        cv2.imwrite(datetime.now().isoformat()+'.jpg', cimg)

    def distance_to_camera(self, frame):
        # KNOWN_DISTANCE = 30.0#meters
        KNOWN_RADIUS = 0.3#meters
        focalLength = 0.00036

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.medianBlur(gray_frame, 5)
        circles = cv2.HoughCircles(gray_frame, cv.CV_HOUGH_GRADIENT, 1, 75)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # compute and return the distance from the maker to the camera
            return (KNOWN_RADIUS * focalLength) / circles[0,2]
        return 30
        

if __name__ == '__main__':
    print __doc__

    # import sys
    # try:
    #     video_src = sys.argv[1]
    # except:
    #     video_src = 0
    App().run()
