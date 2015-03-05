from __future__ import division
import numpy as np
import cv2
import cv2.cv as cv
import video

class App:


    def __init__(self, src, doRecord=True, showWindows=True):
        self.doRecord = doRecord
        self.show = showWindows
        self.frame = None

        self.cap = video.create_capture(src)
        self.cap.set(3,1280)
        self.cap.set(4,2316)
        self.ret, self.frame = self.cap.read() #Take a frame to init recorder
        self.isRecording = False

        self.trigger_time = None
        if showWindows:
            cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
            # cv2.createTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)

    def run(self):
        dist = 30 # meters
        speed = 100 # meters/second
        print "started"
        while True:
            ret, frame = self.cap.read()
            currentframe = frame.copy()
            if self.show:
                cv2.imshow("Image", currentframe)

            if self.trigger_time:
                if self.cap.get(0)==self.trigger_time +time_interval:
                    self.processImage(currentframe)

            c = cv2.waitKey(1) % 0x100
            if c==10: #Catch the current frame timestamp when press 'enter'
                #TODO: Handle differently when the video is recording real-time
                time_interval = dist * 1000 / speed
                self.trigger_time = self.cap.get(0)
            elif c==27: #Break if user enters 'Esc'.
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

        circles = cv2.HoughCircles(gray_frame, cv.CV_HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
        if  circles is not None:
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                if (i[0] - center_x)*(i[0] - center_x) + \
                    (i[1] - center_y)*(i[1] - center_y) < \
                    i[2]*i[2]:
                    print('You have a hit')
                else:
                    print('You missed')
                # draw the center of the circle
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        else:
            print('You missed')
        cv2.imshow("detected circles", cimg)

if __name__ == '__main__':
    print __doc__

    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src, ).run()
