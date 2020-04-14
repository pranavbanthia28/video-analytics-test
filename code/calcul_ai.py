#!/usr/bin/env python
# coding: utf-8

#  ### RealValue.ai  -- Laundry Camera Review assessment 

# In[1]:


import numpy as np
import pandas as pd
import cv2                              # openCV
import requests                         # Sending and receiving http request from/to the api
from pprint import pprint               # Pretty printing JSON 
import os
import base64                           # igame_api needs a base 64 encoded image
import plotly.graph_objects as go       # PLotting time series analysis
import matplotlib.pyplot as plt         # PLotting line charts
import datetime                         # For generating and storing timestamps
import pytz                             # For converting timestamps to relevant timezone
from pytz import timezone


# ### Step 1 - We first load the video into our python program and extract necessary frames from it.

# In[2]:


#########
# Video is loaded into the cv2 object created here and frames are extracted from the same. 
# Saving the extracted frames back in the output_frames directory, solely for the purpose of monitoring
# saving frames is optional, we can remove that line and instead directly pass frame to the next step.

# @Library used   :  openCV
#########

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)   # Current position of the video file in milliseconds
    hasFrames,image = vidcap.read()
    path = 'C://Users//Pranav Banthia//Realvalue//output_frames'
    if hasFrames:
        cv2.imwrite(os.path.join(path,"frame_"+str(sec)+"_sec.jpg"), image)     # save frame as JPG file
    return hasFrames                                         # true if operation was successful


# below lines can be used for video statistics
# fps = vidcap.get(cv2.CAP_PROP_FPS)      
# frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
# duration = frame_count/fps
# print('fps = ' + str(fps))
# print('number of frames = ' + str(frame_count))
# print('duration (S) = ' + str(duration))
# minutes = int(duration/60)
# seconds = duration%60
# print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))


# ### Step 2 - Read a frame and convert it to base64 encoding

# In[3]:


#########
# Frame is loaded and converted to a base64 encoded image since the realvalue igame api needs this as an input.

# @Library used     : base64

# @Param  imageBase64 : stores the base64 image, which we return after utf-8 decoding 
#########

def convert_image_base64(frame_name):
    with open(frame_name, "rb") as img_file:
        imageBase64 = base64.b64encode(img_file.read())
    
    return imageBase64.decode('utf-8')        # Remove the character b'' from the start of the string (b'' gets appended automatically as a part of base64 operation) 


# ### Step 3 - (Approach 1) Call the realvalue igame_api which takes the output from step 2

# In[4]:


#########
# Igame_api takes a base 64 encoded image and the api key as parameter input. Returns the json output with the gender, age, emotion etc

# @Library used   : requests

# @Param     api_key : api_key associated with my username
# @Param    endpoint : api endpoint for our http post request
# @Param imageBase64 : base64 encoded image from the previous step

#########

# Constants
api_key  = '5a105273-70eb-4d3d-afe1-b07bd0346860'
endpoint = "https://massive-plasma-218922.appspot.com/igame_api"

def call_igame_api(imageBase64):
    # json input to api passed as a parameter
    parameters = {  
                    "images": imageBase64, 
                   "api_key": api_key        
                 }
    
    response = requests.post(endpoint, data = parameters) # Send a http post request as per API documentation 
    
    result_body = response.json()
    #pprint(result_body)
    return result_body


# ### (Approach 2) Haar Cascade openCV image classifier 
# #### Haar cascade algorithm works on images to find the number of faces. Aim is to compare the accuracy of the detected number of people with the Real Value igame_api 

# In[5]:


########
# OpenCV Haar Cascade algorithm to analyze images -- Approach 2
# Returns the number of faces it finds in the image passed to the function
########

def haar_cascade_classifier(image_name):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Default file needed for haar cascade to run
    image = cv2.imread(image_name)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImage)
    try:
        return faces.shape[0]
    except:
        return 0


# ### Step 4 - Time Series Plot of the above data 

# In[6]:


def plot_time_series(x,y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y,
                        mode='lines',
                        name='lines'))
    fig.show()


# ### Consolidation and execution of all the above steps

# In[7]:


############
# @Param  frameRate : This determines the number of frames that will be extracted from our video. Can be adjusted as per need.
#                     Keeping this too high will create a very large data file which will be difficult to handle. 
# @Param       sec  : A counter to help us iterate through all the frames in a video   
# @Param   success  : boolean (true/false) depending on whether operation was successful or not
# @Param timestamps : Stores the timestamps of the camera recording at the moment when we extract a frame from it.
#                     The time intervals changes dynamically as per the frame rate we select
#                     Currently it will start at system time whenever we run this program. To scale this variable for camera 
#                     located in any time zone we can use python library called Pytz which takes a local timestamp and 
#                     converts it as per any timezone we specify(timezone will be where the camera is located). This being said,
#                     Pytz will come handy only if we are unable to extract time related meta data from the video itself or http request.
# 
# Sample code -- Pytz library
# Convert local time to Paris time
# current_paris_time = current_sys_time.astimezone(timezone('Europe/Paris'))
# current_paris_time

# Sample Code -- browser meta data (works for most IP cameras)
# response = requests.get('http://82.65.5.211:8082/view/viewer_index.shtml?id=9009')
# response.headers.get('Last-Modified')
# 'Tue, 14 Apr 2020 12:47:45 GMT' -- This is the kind of output we get which can be formatted into a datetime object
############

# Execute step 1
videoFile = '''realvalue_test.mp4'''
vidcap = cv2.VideoCapture(videoFile)


# below calculations will define the length of the video file
fps = vidcap.get(cv2.CAP_PROP_FPS)     
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps           # Get the duration of the video for our stopping condition


sec = 0
frameRate = 1.0                     # Frame rate paramter can be adjusted using this variable
success = getFrame(sec)


# To display time series data, generate timestamps of frame capture
current_date = datetime.datetime.now()
dt = datetime.datetime(current_date.year, current_date.month, current_date.day, current_date.hour, current_date.minute)
increment_rate = (1/frameRate)
step = datetime.timedelta(seconds=increment_rate)
timestamps = []

 
while sec<=duration:                # To run in real time, this condition will be changed to always be true
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    timestamps.append(dt.strftime('%H:%M:%S'))   # Storing timestamps for all the frames
    dt += step


# In[9]:


############
# Execute step 2, 3 Approach 1 (Real Value API)
############


num_people = []       #stores the number of people per image at a given time
logs = []             # stores the error logs if any
gender_male = []      # number of males at a given time
gender_female = []    # number of females at a given time

folder = "C://Users//Pranav Banthia//Realvalue//output_frames"
for filename in os.listdir(folder):
    api_result = call_igame_api(convert_image_base64(os.path.join(folder,filename)))
    male = 0
    female = 0
    try:
        num_people.append(len(api_result['data'][0]))
        tmp = api_result['data'][0]
        for x in tmp:
            if(x['gender'] == 'M'):   # Counting the total males / females at a given time
                male+=1
            else:
                female+=1
         
        gender_male.append(male)   # Finally recording the counts 
        gender_female.append(female)
    except:
        logs.append(result_body)


# # Descriptive Analytics

# In[10]:


############
# Execute step 4 Approach 1
# Graph will be much smoother when we have large number of data points
# You can zoom into this graph for more granular details. Select any area over this graph and it will automatically zoom into it.
# Hovering over any point will give its coordinates (number of people, timestamp)
############
    
plot_time_series(timestamps,num_people)


# In[11]:


# Execute step 2, 3 Approach 2 (Haar Cascade Library)
num_people_2 = []       #stores the number of people per image
logs_2 = []             # stores the error logs

for filename in os.listdir(folder):
    hc_result = haar_cascade_classifier(os.path.join(folder,filename))
    try:
        num_people_2.append(hc_result)
    except:
        logs_2.append(hc_result)


# In[12]:


# Execute step 4 Approach 2

plot_time_series(timestamps,num_people_2)


# In[13]:


# Comparing approach 1 and 2
fig = go.Figure()

fig.add_trace(go.Scatter(x=timestamps, y=num_people,
                    mode='lines',
                    name='Real Value API'))

fig.add_trace(go.Scatter(x=timestamps, y=num_people_2,
                    mode='lines',
                    name='OpenCV Haar Cascade Method'))

fig.show()


# In[14]:


# Line chart of count of male vs female 
fig = go.Figure()

fig.add_trace(go.Scatter(x=timestamps, y=gender_male,
                    mode='lines',
                    name='Male'))

fig.add_trace(go.Scatter(x=timestamps, y=gender_female,
                    mode='lines',
                    name='Female'))

fig.show()


# In[34]:


# Video statistics
indices = [i for i, x in enumerate(num_people) if x == max(num_people)]
time = set()
for i in indices:
    date_time_obj = datetime.datetime.strptime(timestamps[i], '%H:%M:%S')
    s = str(date_time_obj.hour) + ':' + str(date_time_obj.minute)
    time.add(s)
    
print('Maximum people seen in the store :', max(num_people))
print('Maximum people detected at time(s): ', time)
print('Male to Female ratio in the store: ', round((sum(gender_male)/sum(gender_female)),1))


# In[33]:


# Average Hourly statistics

average_per_hour = [0.0 for i in range(0,23)] 
timeframes_per_hour = [0 for i in range(0,23)]
for i in range(0,len(num_people)):
    date_time_obj = datetime.datetime.strptime(timestamps[i], '%H:%M:%S')
    average_per_hour[date_time_obj.hour] += num_people[i]
    timeframes_per_hour[date_time_obj.hour] += 1

    
for i in range(0,len(average_per_hour)):
    if(timeframes_per_hour[i]>0):
        average_per_hour[i] = round((average_per_hour[i]/timeframes_per_hour[i]),1)


# Right now this will just show one bar because of the size of the test data
hours = [str(i)+'a' for i in range(1,12)]
hours.append('12p')
hours_p = [str(i)+'p' for i in range(1,12)]
for i in hours_p:
    hours.append(i)
    


fig = go.Figure(data=[go.Bar(x=hours, y=average_per_hour)])
# Customize aspect
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Average number of people per hour',
                  xaxis_tickangle=-45,
                  yaxis=dict(title='Average Count',
                            titlefont_size=16,
                            tickfont_size=14),
                 xaxis=dict(title='Hour of the day',
                            titlefont_size=16,
                            tickfont_size=14))
fig.show()    

