# To use:
# in .py file folder: /Users/opencv/light_detection_stream
# - workon cv
# - deactivate
# - (cv) python light_detection_stream.py -d true (or false)

# Issue list:
# TODO If no light source, keep running feed! Fix this loop
# TODO Make resolution always native to camera
# TODO Add all operation steps - DONE
# TODO Add commandline arguments - DONE

# Imports
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import time
import cv2

MIN_INTENSITY = 225
MAX_INTENSITY = 255

debugMode = None

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", required=True,
	help="Activate debugmode or not with true or false")
args = vars(ap.parse_args())

# Begin video stream, allow the camera sensor to warm up
# Start FPS counter
print("[INFO] Begin video stream...")
print("[INFO] Press 'q' to exit")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
	
# "Pixelate" the frame to 128x72
def interpolate(inputFrame):
	return cv2.resize(inputFrame, (128, 72), interpolation=cv2.INTER_AREA)

# Scale up pixelated frame
def interpolateSizeUp(inputFrame):
	return cv2.resize(inputFrame, (1280, 720), interpolation=cv2.INTER_NEAREST)

# Read arg to global var
def readDebugArg():
		global debugMode
		if args["debug"] == "true":
			debugMode = True
		else:
			debugMode = False
        
# Outputs coordinates in debugmode
def coordinateOutput(i, *c):
	print("x, y coords for blob #", i)
	print(*c, sep = ", ") 
	print("\n")

def thresholdmask(inputFrame):
  # Threshold frame to reveal lighter regions
	threshold = cv2.threshold(inputFrame, MIN_INTENSITY, MAX_INTENSITY, cv2.THRESH_BINARY)[1] 		
	# Perform erosions and dilations in order to remove any smaller blobs of noise from threshold image
	threshold = cv2.erode(threshold, None, iterations=0) 	#2																					
	threshold = cv2.dilate(threshold, None, iterations=0)	#4
	
 	# Show video output of masking (DEBUG)
	if debugMode == True:
		cv2.imshow("Mask", threshold)																																			

	# Perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components
	labels = measure.label(threshold, background=0) 																									
	mask = np.zeros(threshold.shape, dtype="uint8")

	for (i, label) in enumerate(np.unique(labels)):
		# Simple limit to 5 contours
		if i > 6:
			continue

		# If this is the background label, ignore it
		if label == 0:																																								
			continue

		# Otherwise, construct the label mask and count the number of pixels 
		labelMask = np.zeros(threshold.shape, dtype="uint8") 																					
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)

		# If the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
		if numPixels > 1 and numPixels < 2000: 																										
			mask = cv2.add(mask, labelMask)

	# Find the contours in the mask, then sort them from left to right
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)						
	cnts = imutils.grab_contours(cnts)
	cnts = contours.sort_contours(cnts)[0]
  
	return cnts

def fillMaskAddText(cnts, inputFrame):
	for (i, c) in enumerate(cnts):
																					
		cv2.fillPoly(inputFrame, pts = [c], color=(0,128,0))

		if debugMode == True:
			coordinateOutput(i, *c)

	inputFrame = interpolateSizeUp(inputFrame)

	(x, y, w, h) = cv2.boundingRect(c)
	cv2.putText(inputFrame, "#{}".format(i + 1), ((x*10), (y*10) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

	return inputFrame

while True:
  # Read command line debug arguments
	readDebugArg()
  # Read a frame from videostream
	frame = vs.read()			
	# Get camera resolution
	height, width = frame.shape[:2] 																								
	# Make the frame grayscale
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Apply blur to the greyed frame
	blurredFrame = cv2.GaussianBlur(grayFrame, (41, 41), 0)
	# Pixelate frame
	pixellatedFrame = interpolate(blurredFrame)
 	# Detect the brightest regions and return the contours of the frame
	cnts = thresholdmask(pixellatedFrame) 																					
	# Add text and color to countours (DEBUG FLAG)
	pixellatedFrame = fillMaskAddText(cnts, pixellatedFrame)

	# Show frame (DEBUG FLAG)
	if debugMode == True:
		cv2.imshow("Output", pixellatedFrame)
														
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):																						
		break

	fps.update()

# Stop timer, display FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.stop()
