# emotion-detection-raspberry

In this repository we will first apply transfer learning to import the tensorflow Inception model and we will retrain the last layer using the CK+ dataset that we will expand with pictures scraped from Google Image.

The achieved model will then be implemented on a Raspberry Pi 3 B+

Further improvements will consist in:
- handle wait times between new detection
- logging detections in a MySQL db and expose these logs in a RESTful API.
- Using the KDEF dataset in addition to the CK+ dataset for better accuracy.
- add threading and multiprocessing to improve fps as detailed [here](https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/) and [here](https://github.com/datitran/object_detector_app/blob/master/object_detection_multithreading.py)
- daemonize service

## Pre-requisite

Having a Raspberry Pi 3 B+

Install Tensorflow on it

`sudo apt-get install python3-pip python3-dev`

```
wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
sudo pip3 install tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
```

If the above command doesn't work try renaming the wheel file by replacing 34 by 35 (your python version).
Finally if you have mock installed, uninstall then reinstall:

```
sudo pip3 uninstall mock
sudo pip3 install mock
```

Installing OpenCV on a raspberry pi can be a bit tricky.
I followed this tutorial:
[https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/](https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/)
But instead of downloading the 3.3.0 archive I took the 3.4.0
You really have to follow this tuto step by step without error, otherwise OpenCV won't compile. Count approximately 2-3 hours for that.

## Transfer learning

Inception is a TF pre trained model. The training has been done on millions of images classifiying them in 1000 classes. Training such a big model would be time consuming and expensive. That's where transfer learning becomes useful.
The first layers of Inception are basically edges and shape recognition layers. This is a task which will not change whatever we want to recognize. Therefore we can save a lot of training by reusing the weights of these layers and only training our last layers to classify properly what we want to figure out: emotions on faces.

In this retraining we will feed the CK+ dataset to the Inception model the following way:
- clone tensorflow master repository locally
- in tf local repo root folder create a new directory. Mine is called retrained_data.
- copy the CK+ folders (angry, contempt...) in retrained_data/dataset/ and convert pictures in JPG if they are not already in this format
- retrain the model using the following command:

```
python ./tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/retrained_data/bottlenecks \
--how_many_training_steps 4000 \
--model_dir=/retrained_data/inception \
--output_graph=/retrained_data/retrained_graph.pb \
--output_labels=/retrained_data/retrained_labels.txt \
--image_dir /retrained_data/dataset
```

Here we retrain for 4000 epochs. The CK+ dataset is not very big and you might want to augment it.
Other datasets such as FER2013 or KDEF/AKDEF can be used to complement it. Another augmentation technique would be to use distortion, cropping, brightness changes. This can be achieved using the following parameters in your retrain command: `--flip_left_right`, `--random_crop 10`, `--random_scale 10` and `--random_brightness 10`
However these commands make the training much longer since the bottlenecks are no longer reused at each epoch. TF suggests to use these only to polish your model before production.
I also found an interesting article to collect images from Google Image Search:
[https://nycdatascience.com/blog/student-works/facial-expression-recognition-tensorflow/](https://nycdatascience.com/blog/student-works/facial-expression-recognition-tensorflow/)
Though it requires some manual review, it will definitely help diversifying your training examples.

FER2013 can be downloaded on Kaggle: [https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
More datasets can also be found here: [https://www.mmifacedb.eu/collections/](https://www.mmifacedb.eu/collections/)


Now that you have a trained model, connect by sftp to your raspberry and copy the files retrained_graph.pb, retrained_labels.txt from your retrained_data folder into a new folder on the raspberry.

## PiCamera

We are using the official raspberry pi camera v2.1. We need the `picamera` module for python.

A simple test to see if it works:

```import picamera 
import time

with picamera.PiCamera() as camera:
    camera.resolution = (2592, 1944)
    camera.start_preview()
    time.sleep(2)
    camera.exif_tags['IFD0.Copyright'] = 'Copyright (c) 2018'
    camera.capture('./foo.jpg')
    camera.stop_preview()
```
This captures a screenshot 2s after activation of the camera then close the video feed preview.

## OpenCV

We use the HaarCascade Classifier to find out faces in the video stream.
A future optimization would be to use the new OpenCV DNN Classifier.

First we instantiate the picamera and give it time to start
```
# instanciate the camera
camera = PiCamera()
camera.resolution = (1920, 1080)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(1920, 1080))
# allow the camera to warmup
time.sleep(0.1)
```

We then load our labels and the previously trained model:

```
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("./retrained_data/retrained_labels.txt")]
# load our pretrained model
with tf.gfile.FastGFile("./retrained_data/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
```

Then we need to load the HaarCascade Classifier from the opencv models library. I've included the xml to this repo.

'faceDetect = cv2.CascadeClassifier('./retrained_data/haarcascade_frontalface_default.xml')'

Then the streaming loop is quite self-explanatory. We create the tf session, start the streaming in raw format from the picamera, display it, transform to grey scale, use opencv `detectMultiScale` to return the detected faces. We then transform the grayed image to a numpy array and for each face detected feed the cropped data to the tf graph. We then run our prediction and display the highest probability class label.

When the user press 'q' the demo ends.

```
# start the tensorflow session and start streaming and image processing
sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    # show the frame
    cv2.imshow("face", frame)

    # transform to Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in our gray picture
    faces = faceDetect.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=5
                                        )

    # transform into a numpy array for tf processing
    gray_np = gray.array

    for (x,y,w,h) in faces:
        #sampleNum = sampleNum+1
        #cv2.imwrite("./temp_dataset/"+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
        
        # feed the detected face (cropped image) to the tf graph
        predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': gray_np[y:y+h,x:x+w]})
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
```