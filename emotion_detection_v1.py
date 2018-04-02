from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import tensorflow as tf
import sys

# instanciate the camera
camera = PiCamera()
camera.resolution = (1920, 1080)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(1920, 1080))

# allow the camera to warmup
time.sleep(0.1)

# We use the Haar Cascade classifier
faceDetect = cv2.CascadeClassifier('./retrained_data/haarcascade_frontalface_default.xml')

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
 
	# show the frame
	cv2.imshow("face", image)

    # transform to Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in our gray picture
    faces = faceDetect.detectMultiScale(gray,
                                          scaleFactor=1.3,
                                          minNeighbors=5
                                          )

    for (x,y,w,h) in faces:
        #sampleNum = sampleNum+1
        #cv2.imwrite("./temp_dataset/"+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100);

	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		cv2.destroyAllWindows()
        break




"""
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("./retrained_data/retrained_labels.txt")]

with tf.gfile.FastGFile("./retrained_data/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


# feed the 20 tf images to our tensorflow model
i = 1
while ( i <= 20):	
	#gets the images one by one	
	image_path = "./temp_dataset/"+str(i)+".jpg"
	image_data = tf.gfile.FastGFile(image_path, 'rb').read()
	
	with tf.Session() as sess:
    		# Feed the image_data as input to the graph and get first prediction
    		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    		predictions = sess.run(softmax_tensor, \
             		{'DecodeJpeg/contents:0': image_data})
    
    		# Sort to show labels of first prediction in order of confidence
    		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    		for node_id in top_k:
        		human_string = label_lines[node_id]
        		score = predictions[0][node_id]
        		print('%s (score = %.5f)' % (human_string, score))
	i = i +1

"""