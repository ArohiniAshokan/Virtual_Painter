import cv2
import mediapipe as mp
import HandTrackingModule as htm
import numpy as np
import time
from math import sqrt

previous_time = 0
detector = htm.handDetector()

draw_color = (0,0,0)     #just giving a color
brush_size=20
erasor_size=80
shape=" "

#creating canvas for drawing
img_canvas= np.zeros((720,1280,3),np.uint8)   #ht,wdth,channel- RGB    ,storage 8 bit

video=cv2.VideoCapture(0)

while True:
#1. import frames

    success,frame = video.read()

    frame = cv2.flip(frame,1)   #to flip the frames

    frame=cv2.resize(frame,(1280,720))

    cv2.rectangle(frame,pt1=(0,0),pt2=(1280,100),color=(0,0,0),thickness=-4)
    cv2.rectangle(frame,pt1=(10,10),pt2=(150,90),color=(0,0,255),thickness=-4)
    cv2.rectangle(frame,pt1=(160,10),pt2=(310,90),color=(0,255,0),thickness=-4)
    cv2.rectangle(frame,pt1=(320,10),pt2=(470,90),color=(255,0,0),thickness=-4)
    cv2.rectangle(frame,pt1=(480,10),pt2=(630,90),color=(255,255,0),thickness=-4)
    cv2.rectangle(frame,pt1=(640,10),pt2=(790,90),color=(255,0,255),thickness=-4)
    cv2.rectangle(frame,pt1=(800,10),pt2=(950,90),color=(255,255,255),thickness=-4)
    cv2.rectangle(frame,pt1=(960,10),pt2=(1110,90),color=(255,255,255),thickness=-4)
    cv2.rectangle(frame,pt1=(1120,10),pt2=(1270,90),color=(230,230,230),thickness=-4)

    cv2.putText(frame,text="ERASER",org=(1132,60),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,0,0),thickness=2)
    cv2.putText(frame,text="Circle",org=(997,60),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,0,0),thickness=1)
    cv2.putText(frame,text="Rectangle",org=(818,60),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,0,0),thickness=1)


#2. find hand landmarks
    frame = detector.findHands(frame)   #it will detect hands from frames
    lmlist = detector.findPosition(frame)  # lmlist is landmark list, the position coordinates of detected hand from frames is stored as list in lmlist
    #print(lmlist)

    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]  #selecting the coordinates of  tip of index finger
        x2,y2=lmlist[12][1:]  #middle finger
        x0,y0=lmlist[4][1:]
        #print(x1,y1)

#3. find which finger is up
        fingers=detector.fingersUp()  #to detect which finger is up
        #print(fingers)  #this is a list with 0 and 1 , 0 - finger down, 1- finger up 


#4. if two fingers are up then selection mode
        if (fingers[1] and fingers[2]):  #where fingers[1] means index finger and fingers[2] means middle finger
            
            xp,yp = 0,0  # used for drawing

            print('selection mode')
            
            if y1<120:      #if our finger tip is inside the black boundary
                
                if 10<x1<150:      #checking if finger tip is in which color box
                    draw_color=(0,0,255)
                    shape='freestyle'
                
                elif 160<x1<310:
                    draw_color=(0,255,0)
                    shape='freestyle'
                
                elif 320<x1<470:
                    draw_color=(255,0,0)
                    shape='freestyle'
                
                elif 480<x1<630:
                    draw_color=(255,255,0)
                    shape='freestyle'
                
                elif 640<x1<790:
                    draw_color=(255,0,255)
                    shape='freestyle'
                
                elif 1120<x1<1270:
                    draw_color=(0,0,0)

                elif 800<x2<950:
                    shape="rectangle"

                elif 960<x2<1110:
                    shape="circle"

            cv2.rectangle(frame,(x1,y1),(x2,y2),draw_color,-3)   #drawing a rectangle at the finger tips to indicate chosen drawing colors


#5. if one finger is up then drawing mode
        if (fingers[1] and not fingers[2]):
            print('drawing mode')

            cv2.circle(frame,(x1,y1),15,draw_color,thickness=-3)

            if xp==0 and yp==0:
                xp=x1
                yp=y1
            
            if draw_color==(0,0,0):
                cv2.line(frame,(xp,yp),(x1,y1),draw_color,thickness= erasor_size)
                cv2.line(img_canvas,(xp,yp),(x1,y1),draw_color,thickness= erasor_size)
            else:

                if shape=='freestyle':
                    cv2.line(frame,(xp,yp),(x1,y1),draw_color,thickness= brush_size)
                    cv2.line(img_canvas,(xp,yp),(x1,y1),draw_color,thickness= brush_size)

                if shape=='rectangle':

                    cv2.rectangle(frame,(x0,y0),(x1,y1),draw_color,-3)

                    if fingers[4]:
                        cv2.rectangle(img_canvas,(x0,y0),(x1,y1),draw_color,-3)
                        

                if shape=='circle':
                    x3,y3=lmlist[4][1:]
                    result=int(sqrt(((x3-x1)**2)+((y3-y1)**2)))

                    if result<0:
                        result= (-1*result)

                    cv2.circle(frame,(x0,y0),result,draw_color,-3)

                    if fingers[4]:
                        cv2.circle(img_canvas,(x0,y0),result,draw_color,-3)

            xp,yp=x1,y1

#first convert the canvas to grey and then apply threshold to inverse it, then it back to BGR
    canvas_grey = cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)
    _,canvas_inverse = cv2.threshold(canvas_grey,20,255,cv2.THRESH_BINARY_INV)
    canvas_inverse = cv2.cvtColor(canvas_inverse,cv2.COLOR_GRAY2BGR)

    #blending frame and canvas
    frame = cv2.bitwise_and(frame,canvas_inverse)
    frame = cv2.bitwise_or(frame,img_canvas)

    #fps 
    current_time = time.time()
    fps= 1/(current_time- previous_time)
    previous_time=current_time

    cv2.putText(frame,str(int(fps)),(50,300),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
    
    #merging
    frame = cv2.addWeighted(frame,1,img_canvas,0.5,0)   #alpha -1, beta- 0.5, gamma-0

    cv2.imshow('video',frame)
    #cv2.imshow('canvas',img_canvas)
   
    
    if cv2.waitKey(1) & 0XFF==27:
        break
video.release
cv2.destroyAllWindows()