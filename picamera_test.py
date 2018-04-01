import picamera 
import time

with picamera.PiCamera() as camera:
    camera.resolution = (2592, 1944)
    camera.start_preview()
    time.sleep(2)
    camera.exif_tags['IFD0.Artist'] = 'Vincent Teyssier'
    camera.exif_tags['IFD0.Copyright'] = 'Copyright (c) 2018 Vincent Teyssier'
    camera.capture('foo.jpg')
    camera.stop_preview()