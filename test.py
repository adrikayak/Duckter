import numpy as np
import cv2
from datetime import datetime
import time
import video
import scipy.spatial
from common import anorm2, draw_str
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
class Duckter:

    def __init__(self, threshold = 25, doRecord=True, showWindows=True):
        self.doRecord = doRecord
        self.show = showWindows
        self.frame = None
        
        # self.cap = video.create_capture(src)
        # self.cap.set(3,1280)
        # self.cap.set(4,2316)
        # self.ret, self.frame = self.cap.read() #Take a frame to init recorder
        self.frame_rate = camera.framerate
        print self.frame_rate
        self.gray_frame = np.zeros((320,240, 1), np.uint8)
        self.average_frame = np.zeros((320,240, 3), np.float32)
        self.absdiff_frame = None
        self.previous_frame = None
        
        # self.surface = self.cap.get(3) * self.cap.get(4)
        # self.currentsurface = 0
        self.currentcontours = None
        self.threshold = threshold
        self.isRecording = False

        self.tracks = []
        self.tracks_dist = []
        self.track_len = 3
        self.frame_idx = 0
        self.detect_interval = 5      
        # self.font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font
        self.trigger_time = 0
        if showWindows:
            cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
            # cv2.createTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)

    def run(self):
        started = time.time()
        print "started"
        print started

        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            self.frame = frame.array
         
            # # show the frame
            # cv2.imshow("Frame", image)

            currentframe = frame.array
            # cv2.ims	e", currentframe)
            instant = time.time()
            # print instant
            self.processImage (currentframe)
            if not self.isRecording:
                if self.somethingHasMoved():
                    self.speedEstimation()
                cv2.drawContours(currentframe, self.currentcontours,-1,(0, 0, 255),2)
            if self.show:
                for dist in self.tracks_dist:
                    if dist[2] > 0:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # cv2.putText(currentframe,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
                        # cv2.putText(currentframe, str(dist[2]/(9*5/30)), (60, 60), font, 4,(255,255,255),2,cv2.CV_AA)
                        draw_str(currentframe,(dist[0],dist[1]), str(dist[2]/(9*5/30)))
                cv2.imshow("Image", currentframe)
            self.prev_gray = self.gray_frame
            self.frame_idx += 1
            rawCapture.truncate(0)
            c = cv2.waitKey(1) % 0x100
            if c==27 or c == 10: #Break if user enters 'Esc'.
                break
            # key = cv2.waitKey(1) & 0xFF
         
            # clear the stream in preparation for the next frame
         
            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
                # break


    def processImage(self, curframe):
        # cv.Smooth(curframe, curframe) #Remove false positives
        curframe = cv2.GaussianBlur(curframe,(5,5),0)
        # GaussianBlur(gray_image, canny_image, Size(17, 17), 2, 2);
        if self.absdiff_frame == None: #For the first time put values in difference, temp and moving_average
            self.absdiff_frame = curframe.copy()
            self.previous_frame = curframe.copy()
            self.average_frame = np.float32(curframe) #Should convert because after runningavg take 32F pictures
            # cv2.imshow("average_frame",self.average_frame)
        else:
            cv2.accumulateWeighted(curframe, self.average_frame, 0.05) #Compute the average
            # cv.RunningAvg(curframe, self.average_frame, 0.05) #Compute the average
        self.previous_frame = np.uint8(self.average_frame) #Should convert because after runningavg take 32F pictures
        # cv2.imshow("previous_frame", self.previous_frame)

        cv2.absdiff(curframe, self.previous_frame, self.absdiff_frame) # moving_average - curframe
        # self.absdiff_frame = self.watershed(self.absdiff_frame)
        self.gray_frame = cv2.cvtColor(self.absdiff_frame, cv2.COLOR_BGR2GRAY) #Convert to gray otherwise can't do threshold
        # ret, self.gray_frame = cv2.threshold(self.gray_frame,50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, self.gray_frame = cv2.threshold(self.gray_frame, 50, 255, cv2.THRESH_BINARY)
        self.gray_frame = cv2.dilate(self.gray_frame, None, 15) #to get object blobs
        self.gray_frame = cv2.erode(self.gray_frame, None, 10)

    def somethingHasMoved(self):
        
        # Find contours
        # image, contours, hierarchy = cv2.findContours(self.gray_frame, 1, 2)
        # cv2.imshow("gray_frame",self.gray_frame)
        contours, hierarchy = cv2.findContours(self.gray_frame, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[0]
        self.currentcontours = contours #Save contours
        
        # if contours: #For all contours compute the area
        #     self.currentsurface += cv2.contourArea(contours[0])
        # avg = (self.currentsurface*100000)/self.surface #Calculate the average of contour area on the total size
        # self.currentsurface = 0 #Put back the current surface to 0
        # print avg
        # if avg > self.threshold:
        #     print "true"
        return True

    def speedEstimation(self):
        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(self.gray_frame)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
                # cv2.imshow("mask",mask)
            p = cv2.goodFeaturesToTrack(self.gray_frame, mask = mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])

        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, self.gray_frame
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            self.tracks_dist = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                dist = 0
                if len(tr) > self.track_len:
                    del tr[0]
                    XA =  np.reshape(tr[:9],(-1,2))
                    XB =  np.reshape(tr[1:],(-1,2))
                    eu_dists = scipy.spatial.distance.cdist(XA, XB, 'euclidean')
                    for eu_dist in eu_dists:
                        dist += eu_dist[0]
                self.tracks_dist.append ([int(x), int(y), round(dist,2)])
                # print self.tracks_dist
                # print dist
                new_tracks.append(tr)
                # cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            self.tracks = new_tracks
            # print len(self.tracks[0])
            # cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

if __name__ == '__main__':
    print __doc__
    Duckter().run()
