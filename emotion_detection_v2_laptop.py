from picamera.array import PiRGBArray
from picamera import PiCamera
from multiprocessing import Process
from multiprocessing import Queue
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


def prediction_fn(model, inputQueue, outputQueue, label_lines):
    while True:
        if not inputQueue.empty():
            start1 = time.time()
            # grab the frame from the input queue			
            input_image = inputQueue.get()
			
			# make the prediction
			# feed the detected face (cropped image) to the tf graph
            predictions = sess.run(model, {'DecodeJpeg:0': input_image})
            prediction = predictions[0]

            # Get the highest confidence category.
            prediction = prediction.tolist()
            max_value = max(prediction)
            max_index = prediction.index(max_value)
            predicted_label = label_lines[max_index]

            end1 = time.time()
            print("%s (%.2f%%) %.4f" % (predicted_label, max_value * 100, end1-start1))
			# write the detections to the output queue
            outputQueue.put(predicted_label)

# start the tensorflow session and start streaming and image processing
sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

# initialize the input queue (detected faces grayed), output queue (predictions),
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
predictions = None

# initialize the prediction process
p = Process(target=prediction_fn, args=(softmax_tensor, inputQueue,
	outputQueue,label_lines,))
p.daemon = True
p.start()

i=0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    i = i + 1
    start = time.time()
    
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
    end = time.time()
    # printing face detect execution time
    if (i<10):
        print ("Frame detection time: " + (end - start))

    for (x,y,w,h) in faces:
        # if our input queue is empty we pile one detected face for prediction
        if inputQueue.empty():
            inputQueue.put(gray[y:y+h,x:x+w])

        # if there is a prediction in the output queue, we grab it
        if not outputQueue.empty():
            predictions = outputQueue.get()

        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100)

    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()
        break



