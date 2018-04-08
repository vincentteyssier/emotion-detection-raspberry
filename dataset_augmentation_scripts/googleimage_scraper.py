import selenium
import json
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
driver = webdriver.Chrome(r"C:\Program Files\Google\chromedriver.exe")
import os
import urllib

query = input("Search Criteria: ")# you can change the query for the image  here
image_type="ActiOn"
query= query.split()
query='+'.join(query)
#add the directory for your image here
DIR="Pictures"

driver=webdriver.Chrome()
driver.get('https://images.google.com/')
elem=driver.find_element_by_name('q')
elem.send_keys(query)
elem.send_keys(Keys.RETURN)

#scroll down

SCROLL_PAUSE_TIME = 10
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            more=driver.find_element_by_id('smb')
            more.click()
        except:
            break
    last_height = new_height

#Get Pictures
pics=driver.find_elements_by_class_name("rg_meta")

ActualImages=[]

for i in driver.find_elements_by_class_name("rg_meta"):
    link , Type = json.loads(i.get_attribute('innerHTML'))['ou'] , json.loads(i.get_attribute('innerHTML'))['ity']
    ActualImages.append((link,Type))

print ('Total' +len(ActualImages)+ 'Images')

header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}

if not os.path.exists(DIR):
            os.mkdir(DIR)
DIR = os.path.join(DIR, query.split()[0])

if not os.path.exists(DIR):
            os.mkdir(DIR)
###print images
for i , (img , Type) in enumerate( ActualImages):
    try:
        req = urllib.Request(img, headers={'User-Agent' : header})
        raw_img = urllib.urlopen(req).read()

        cntr = len([i for i in os.listdir(DIR) if image_type in i]) + 1
        print (cntr)
        if len(Type)==0:
            f = open(os.path.join(DIR , image_type + "_"+ str(cntr)+".jpg"), 'wb')
        else :
            f = open(os.path.join(DIR , image_type + "_"+ str(cntr)+"."+Type), 'wb')


        f.write(raw_img)
        f.close()
    except Exception as e:
        print ("could not load : "+img)
        print (e)

# preprocess images downloaded above:
Facial_Data=[]
director=[]
filename=[]
Id=[]
directory='Pictures'
n=0
for root, dirs, files in os.walk(directory):
    for direc in dirs:
        director.append('Pictures/'+ direc)

for i in xrange(len(director)):
    for root2, dirs2, files2 in os.walk(director[i]):
        filedirec=map(lambda x : '/'.join([root2,x]),files2)
        filename += filedirec
        Id += [1*n]*len(filedirec)
        n=n+1
        
#only support jpg or jpeg file
filename =[ X for X in zip(filename,Id) if X[0].endswith('.jpg') or X[0].endswith('.jpeg') ]

#Extract Face
import sys

import dlib
from skimage import io
import pickle

Face_Data=[]
Face_Id=[]
n=0
detector = dlib.get_frontal_face_detector()

for pic,ind in filename:
    img = io.imread(pic)
    
    try:
        dets = detector(img, 1)
        print (pic + " " + n)
        for i, d in enumerate(dets):
            face=np.asarray(img[d.top():d.bottom(),d.left():d.right()],order='C')
            Face_Data.append(face)
            Face_Id.append(ind)
    except:
        print ('Fail '+n)
    
    n=n+1
#pickle.dump(Face_Data,open('Face_Data.p','wb'))

#grey
grey_face=[]
grey_face_id=[]
for i in xrange(len(Face_Data)):
    if len(Face_Data[i].shape)==3:
        grey_face.append(color.rgb2grey(Face_Data[i]))
        grey_face_id.append(Face_Id[i])
    elif len(Face_Data[i].shape)==2:
        grey_face.append(Face_Data[i])
        grey_face_id.append(Face_Id[i])

#resize
resized_grey_face=[]
resized_grey_face_id=[]
for j in xrange(len(grey_face)):
    try:
        resized_grey_face.append(resize(grey_face[j],(100,100),mode='constant'))
        resized_grey_face_id.append(grey_face_id[j])
    except:
        print (j)
        
pickle.dump(resized_grey_face,open('resized_grey_face.p','wb'))
pickle.dump(resized_grey_face_id,open('resized_grey_face_id.p','wb'))