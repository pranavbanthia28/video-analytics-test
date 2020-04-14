# Laundry Store Analytics
#### Steps to run the program:
1) download the calcul_ai.ipynb or calcul_ai.py (both are the same files with different extensions).
2) download the haarcascade_frontalface_default.xml file too and save it in the same directory as the code.
3) save the test video provided or any video of your choice in the same directory as the code
4) lastly create a empty directory named output_frames inside the folder where your code is downloaded (this folder stores the frames that we create). Change the path variable and folder variable in the code to the appropriate directory path. Currently, these values are hardcoded with my local path. 

#### Explanation
Most of the documentation is included in the code itself as comments.
I have tried to create a small pipeline for the entire video analytics process. Currently, this works for any test video(saved in the same local directory as your code) whose name you pass as a parameter to the code.

Step 1 - We extract the frames from the video and save it in the output_frames directory.The number of frames to be extracted can be adjusted using the variable frameCount. More details written inside the python file.

Step 2 - Our code reads all the images iteratively from the output_frames directory and converts it to base64 encoding. The part where we save the frames can be eliminated and we can directly pass frames from step 1 to step 2. Currently for test purpose I have kept it as it is. When we have too many frames to process, we can avoid the process of saving and then again reading.

Step 3 - We make the API call to igame using a http post request as per the API documentation. Passed the APi key and the base 64 encoded image. Retrieve the json reply and use that for step 4. The output of our program depends on two factors : 
First - accuracy of the values returned by the API
second - the selected frame rate

Step 4 - We perform descriptive analytics on the data. Just for demonstration purpose I have also used another method of face detection which is provided by the openCV python library. I have created a comparison plot which shows the differences in results between api values and haar cascade. 

#### Possible areas for future work
Some of the key areas where the work can be continued is :
-- I have been able to make a html page which displays a live stream of any IP camera. We can take this page ahead by allowing live video input from the same live IP camera source to our python program and displaying all results on the web page.
-- Provisions have been made to consider camera in any timezone and storing the timestamps accordingly. Comments provided in the code.


#### Other info
The algorithms used by haar cascade or the igame api are trained to detect faces. There are instances like camera blind spots in a room or when people are waiting in corners and the current camera positioning only captures their legs or lower body. In such cases algorithms mostly fails to detect the person.
For example - The person in the top right corner in the image below

![Blind Spot](/screenshot.png)
 
