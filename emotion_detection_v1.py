from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import tensorflow as tf
import sys

DEBUG = True

# instanciate the camera
camera = PiCamera()
camera.resolution = (1920, 1080)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(1920, 1080))

# allow the camera to warmup
time.sleep(0.1)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("./retrained_data/retrained_labels.txt")]
# load our pretrained model
with tf.gfile.FastGFile("./retrained_data/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# We use the Haar Cascade classifier
faceDetect = cv2.CascadeClassifier('./retrained_data/haarcascade_frontalface_default.xml')

# start the tensorflow session and start streaming and image processing
sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    # transform into a numpy array
    image = frame.array
    # show the frame
    cv2.imshow("face", image)
    if DEBUG:
        print (image.shape)
    # transform to Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        print (gray.shape)
    # detect faces in our gray picture
    faces = faceDetect.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=5
                                        )


    for (x,y,w,h) in faces:
        #sampleNum = sampleNum+1
        #cv2.imwrite("./temp_dataset/"+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
        
        # feed the detected face (cropped image) to the tf graph
        predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': gray[y:y+h,x:x+w]})
        prediction = predictions[0]

        # Get the highest confidence category.
        prediction = prediction.tolist()
        max_value = max(prediction)
        max_index = prediction.index(max_value)
        predicted_label = label_lines[max_index]

        print("%s (%.2f%%)" % (predicted_label, max_value * 100))

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
       		predictions = sess.run(softmax_tensor, \
             		{'DecodeJpeg/contents:0': image_data})
    
    		# Sort to show labels of first prediction in order of confidence
    		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    		for node_id in top_k:
        		human_string = label_lines[node_id]
        		score = predictions[0][node_id]
        		print('%s (score = %.5f)' % (human_string, score))
"""