import libjevois as jevois
import cv2
import numpy
import json

class FrcRobot:

    # cubes
    class Cube:
        def __init__(self):
            self.x = 0
            self.y = 0
            self.w = 0
            self.h = 0
            self.track = 0
            self.angle = 0
            self.distance = 0

        def compute(self, width, focalLength, actualDimension, cameraDisplacement, armDisplacement):
            if self.track == 1:
                return

            # angle and distance from camera (angles in radians)
            cA = numpy.arctan2(self.x + self.w / 2 - width / 2, focalLength)
            cD = actualDimension * focalLength / self.w

            # angle and distance after accounting for camera displacement (angles in radians)
            self.track = 1
            self.angle = numpy.arctan2(cD * numpy.sin(cA) + cameraDisplacement, cD * numpy.cos(cA))
            self.distance = cD * numpy.cos(cA) / numpy.cos(self.angle) - armDisplacement

        def toJson(self):
            pixels = {"Track" : 0, "Angle" : 0, "Range" : 0}
            if (self.track == 1):
                pixels = {"Track" : 1, "Angle" : int(numpy.degrees(self.angle)), "Range" : int(self.distance)}
            return json.dumps(pixels)

    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # yellow objects hsv range
        self.lower_yellow = numpy.array([20,200,100])
        self.upper_yellow = numpy.array([40,255,255])

        # all distance dimensions are in inches
        self.cameraDisplacement = 11.25
        self.armDisplacement = 15
        self.actualDimension = 17.0
        self.actualDistance = 82
        self.pixelDimension = 78
        self.focalLength = self.pixelDimension * self.actualDistance / self.actualDimension

        self.minArea = 2000
        self.maxArea = 120000
        self.minAspectRatio = 0.7
        self.maxAspectRatio = 1.4
        self.minExtent = 0.6

        self.recentAngles = [0, 0, 0, 0]
        self.recentDistances = [0, 0, 0, 0]
        self.timer = jevois.Timer("FrcRobot", 100, jevois.LOG_INFO)

    # ###################################################################################################
    ## Load camera calibration from JeVois share directory
    def loadCameraCalibration(self, w, h):
        cpf = "/jevois/share/camera/calibration{}x{}.yaml".format(w, h)
        fs = cv2.FileStorage(cpf, cv2.FILE_STORAGE_READ)
        if (fs.isOpened()):
            self.camMatrix = fs.getNode("camera_matrix").mat()
            self.distCoeffs = fs.getNode("distortion_coefficients").mat()
            jevois.LINFO("Loaded camera calibration from {}".format(cpf))
        else:
            jevois.LFATAL("Failed to read camera parameters from file [{}]".format(cpf))

    # ###################################################################################################
    ## Detect objects within our HSV range
    def detect(self, src, outimg = None):
        cubes = []
        cube_distances = []
        height, width, _ = src.shape

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        yellow = cv2.bitwise_and(src, src, mask = mask)
        gray = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fltr = cv2.bilateralFilter(blur, 1, 10, 100)
        canny  = cv2.Canny(fltr, 10, 100)
        dilation = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, numpy.ones((5,5), numpy.uint8))

        _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for ct in contours:
            area = cv2.contourArea(ct)
            if area > self.minArea and area < self.maxArea:
                # contour properties
                x,y,w,h = cv2.boundingRect(ct)
                area = cv2.contourArea(ct)
                #extent = float(area) / (w*h)
                aspect_ratio = float(w)/h

                if (aspect_ratio < self.maxAspectRatio and aspect_ratio > self.minAspectRatio): # and extent > self.minExtent):
                    cube = self.Cube()
                    cube.x = x
                    cube.y = y
                    cube.w = w
                    cube.h = h
                    cubes.append(cube)
                    cube_distances.append(area)

        if len(cube_distances) > 0:
            cube = cubes[numpy.argmax(cube_distances)]
            cube.compute(width, self.focalLength, self.actualDimension, self.cameraDisplacement, self.armDisplacement)
            self.smooth(cube)
            self.printOnImage(outimg, cube, width, height)
            return cube
        else:
            return None

    def printOnImage(self, outimg, cube, width, height):
        if outimg is not None and outimg.valid() and cube is not None:
            jevois.drawRect(outimg, cube.x, cube.y, cube.w, cube.h, 2, jevois.YUYV.MedPurple)
            jevois.writeText(outimg, "Angle: " + str(int(numpy.degrees(cube.angle))) + 
                " Distance: " + str(int(cube.distance)) +
                " Pixel width: " + str(cube.w),  
                3, height+1, jevois.YUYV.White, jevois.Font.Font6x10)

    def smooth(self, cube):
        self.recentAngles.pop(0)
        self.recentDistances.pop(0)
        self.recentAngles.append(cube.angle)
        self.recentDistances.append(cube.distance)
        cube.angle = numpy.mean(self.recentAngles)
        cube.distance = numpy.mean(self.recentDistances)

    # ###################################################################################################
    ## Process function with no USB output
    def processNoUSB(self, inframe):
        inimg = inframe.getCvBGR()
        cube = self.detect(inimg)
        # Load camera calibration if needed:
        # if not hasattr(self, 'camMatrix'): self.loadCameraCalibration(w, h)
        if cube is not None:
            jevois.sendSerial(cube.toJson())

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        inimg = inframe.get()
        self.timer.start()

        imgbgr = jevois.convertToCvBGR(inimg)
        h, w, chans = imgbgr.shape
        outimg = outframe.get()
        outimg.require("output", w, h + 12, jevois.V4L2_PIX_FMT_YUYV)
        jevois.paste(inimg, outimg, 0, 0)
        jevois.drawFilledRect(outimg, 0, h, outimg.width, outimg.height-h, jevois.YUYV.Black)
        inframe.done()
        
        cube = self.detect(imgbgr, outimg)
        # Load camera calibration if needed:
        # if not hasattr(self, 'camMatrix'): self.loadCameraCalibration(w, h)

        if cube is not None:
            jevois.sendSerial(cube.toJson())

        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        jevois.writeText(outimg, fps, 3, h-10, jevois.YUYV.White, jevois.Font.Font6x10)
        outframe.send()
