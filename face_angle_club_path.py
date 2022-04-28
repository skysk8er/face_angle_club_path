import cv2
import numpy as np
from matplotlib import pyplot as plt
# import natsort
import os
from scipy.spatial import distance
from scipy.stats import mode
import math
from math import atan2, pi
from collections import deque
import pyshine as ps



def cal_dist(pt1,pt2 ,target):
    x1,y1=pt1[0],pt1[1]
    x2,y2=pt2[0],pt2[1]
    a,b= target[0],target[1]
    area = abs((x1-a) * (y2-b) - (y1-b) * (x2 - a))
    AB = ((x1-x2)**2 + (y1-y2)**2) **0.5
    distance = area/AB
    return distance


def preprocessing(frame):
    global img
    img = frame
    # img=cv2.resize(img,(int(img.shape[1]*0.5),int(img.shape[0]*0.5)),interpolation=cv2.INTER_LINEAR)
    blurred=cv2.GaussianBlur(img,(1,1),0)
    # blurred = cv2.medianBlur(img,3)
    

    # 그레이 스케일로 변환 ---①
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #line detection용 
    b_imgray=cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) #club head용

    # 스레시홀드로 바이너리 이미지로 만들어서 검은배경에 흰색전경으로 반전 ---②
    ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY) #line detection용 
    ret2, rf = cv2.threshold(b_imgray, 200, 255, cv2.THRESH_BINARY) #club head용


    # 가장 바깥쪽 컨투어에 대해 모든 좌표 반환 ---③
    contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
    head, hier = cv2.findContours(rf, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)

    return img, contour, head

def bbox(bbox_dir):
    txt_list=os.listdir(dir)
    txt_list = natsort.natsorted(txt_list)
    file = open(box_root+'/'+txt_list[347],'r')
    strings = file.readlines()
    lst=strings[1].split()

    x,y = float(lst[1]),float(lst[2])
    w,h = float(lst[3]),float(lst[4])
    X,W=x * img.shape[1],w * img.shape[1]
    Y,H= y * img.shape[0], h * img.shape[0]

    x1= int(X - W * 0.5)
    y1 = int(Y - H * 0.5)
    x2 = int(X + W * 0.5)
    y2 = int(Y + H * 0.5)

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)

def centerline_detection(contour):
    points=[]
    max_line=[]
    for i in contour:
        
        for j in i:
            if (int(img.shape[1])//3 <= j[0][0] <= int(img.shape[1])//3 *2 ):  
                max_line.append(j[0][0])
    coord1=max(max_line ,key=max_line.count)

    while coord1 in max_line:
        max_line.remove(coord1)
    coord2=max(max_line ,key=max_line.count)
    line_center=int((coord1+coord2)/2)
    
    cv2.line(img,(line_center,0),(line_center,int(img.shape[0])),(0,0,255),2)
    cv2.circle(img, (line_center,0), 1, (0,255,  0),5)

    return line_center


def clubhead_detection(head):
    field=head[0]

    hs=[]
    ws=[]
    for k in field:
        hs.append(k[0][0])
        ws.append(k[0][1])

    low_h,max_h =min(hs), max(hs)
    low_w, max_w =min(ws), max(ws)

    top_idx=np.int0(np.where(ws==low_w))[0][0]
    left_idx=np.int0(np.where(hs==low_h))[0][0]
    right_idx=np.int0(np.where(hs==max_h))[0][0]

    
    top_point=(hs[top_idx],low_w)
    left_point=(low_h,ws[left_idx])
    right_point=(max_h,ws[right_idx])
    
    return top_point, left_point, right_point

def closed_or_open(top_point,left_point,right_point):    

    closed='closed'

    ld, rd=distance.euclidean(top_point,left_point),distance.euclidean(top_point,right_point)
    if ld > rd:
        cv2.line(img,top_point,left_point,(255,0,0),2)
    elif ld==rd:
        closed='square'

    else:
        cv2.line(img,top_point,right_point,(255,0,0),2)
        closed='open'

    return closed

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return (int(x), int(y))


def getAngle(a, b, c):
    """
    b : intersection points
    """
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    
    return round(90-abs(ang),1) if abs(ang) > 45 else round(abs(ang),1)
    

def face_angle(ball,closed,top_point,left_point,right_point,line_center):

    ball_center=(ball[0],ball[1])
    ball_radius=ball[2]
    
    if closed=='closed' :
        dist=cal_dist(left_point,top_point,ball_center)
        if dist < ball_radius:
            intersection=line_intersection((top_point,left_point),((line_center,0),(line_center,int(img.shape[0]))))
            angle=getAngle(top_point,intersection,(line_center,0))
    elif closed=='squared':
        angle=int(0)

    else:
        dist=cal_dist(right_point,top_point,ball_center)
        if dist < ball_radius:
            intersection=line_intersection((top_point,right_point),((line_center,0),(line_center,int(img.shape[0]))))
            angle=getAngle(top_point,intersection,(line_center,0))
        
    if dist < ball_radius:
        cv2.circle(img, intersection, 1, (0,255, 0),5)
        cv2.circle(img, top_point, 1, (0,255, 0),5)
        cv2.line(img,(0,intersection[1]),(int(img.shape[0]),intersection[1]),(255,0,0),1)
        # cv2.putText(img,f'Face Angle : {angle} degree {closed}',(img.shape[0]//4,img.shape[1]//3),1,0.7,(255,255,255),1)
        txt=f'Face Angle : {angle} degree {closed}' 
        ps.putBText(img,txt,text_offset_x=img.shape[1]//4,text_offset_y=img.shape[0]//4,vspace=10,hspace=10, \
            font_scale=img.shape[0]/1920,background_RGB=(228,225,222),text_RGB=(1,1,1))
        
        # cv2.imshow('Impact frame', img)

    return (angle,intersection)  if dist < ball_radius else (None,None)



def mouse_callback(event, x, y, flags, param): 
    print("마우스 이벤트 발생, x:", x ," y:", y) # 이벤트 발생한 마우스 위치 출력

    cv2.namedWindow('image')  #마우스 이벤트 영역 윈도우 생성
    cv2.setMouseCallback('image', mouse_callback)

    while(True):

        cv2.imshow('image', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:    # ESC 키 눌러졌을 경우 종료
            break
    cv2.destroyAllWindows()

def ball_detection(img):

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred= cv2.medianBlur(imgray,5)

    circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=50)
    if circles is None:
        pass
    else:
        circles = np.uint16(np.around(circles))
        dist=[]
        for k,i in enumerate(circles[0,:]):
                dist.append(distance.euclidean((i[0],i[1]),(int(img.shape[1]*0.5),int(img.shape[0]*0.5))))

        ball=circles[0,:][np.argmin(dist)]

    # cv2.circle(img,(ball[0],ball[1]),ball[2],(0,255,0),2)

    return ball if circles is not None else None


def club_path_ratio(top_point,left_point,right_point, intersection,closed):
    
    dist=distance.euclidean(top_point,intersection)
    if closed=='closed':
        total_dist=distance.euclidean(top_point,left_point)
        ratio=dist/total_dist
    elif closed=='square':
        total_dist=distance.euclidean(right_point,left_point)
        ratio=dist/total_dist
    else  :
        total_dist=distance.euclidean(top_point,right_point)
        ratio=dist/total_dist
    
    return round(ratio,2)

def club_path_points(deq,ratio,closed):
    for i in range(len(deq)):

        (x1,y1),(x2,y2) = deq[i]
        if closed =='closed':
            (a,b)=(x1-int(abs(x1-x2)*ratio)),(y1+int(abs(y1-y2)*ratio))
        else:
            (a,b)=(x1+int(abs(x1-x2)*ratio)),(y1+int(abs(y1-y2)*ratio))

        cv2.circle(img,(a,b),1,(0,0,0),2)
        if i==0:
            return (a,b)
        

def club_path(club_path_point,intersection,line_center):
    angle=getAngle(club_path_point,intersection,(line_center,int(img.shape[0])))
    if club_path_point[0] < line_center :
        direction='In-To-Out'
    else:
        direction='Out-To-In'
    # cv2.putText(img,f'Club Path : {angle} degree {direction}',(img.shape[0]//4,img.shape[1]//4),1,0.7,(255,255,255),1)
    txt=f'Club Path : {angle} degree {direction}'
    ps.putBText(img,txt,text_offset_x=img.shape[1]//4,text_offset_y=img.shape[0]//5,vspace=10,hspace=10, \
            font_scale=img.shape[0]/1920,background_RGB=(228,225,222),text_RGB=(1,1,1))

    return direction

def main():
    cap = cv2.VideoCapture('closed.mov')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)
    out = cv2.VideoWriter('output.avi', fourcc, fps, (int(width), int(height)))
    cnt=0
    init_ball=[]
    deq=deque(maxlen=20)
    while(cap.isOpened()):
        ret, frame = cap.read()

        img, contour, head = preprocessing(frame)
        top_point, left_point, right_point = clubhead_detection(head)
        closed =closed_or_open(top_point,left_point,right_point)

        if closed=='closed':
            deq.append((top_point,left_point))
        else:
            deq.append((top_point,right_point))

        line_center=centerline_detection(contour)
        
        if cnt<=10:
            ball=ball_detection(img)
            init_ball.append(ball)
        else: 
            init_ball=[x for x in init_ball if x is not None]
            lst=np.array([(line_center-3 < i[0] < line_center+3) and (img.shape[0]//3 < i[1] <img.shape[0]//3 *2) for i in init_ball])
            idx=np.where(lst==True)[0]
            candidates=[init_ball[i] for i in idx]
            ball=(mode([i[0] for i in candidates])[0][0],mode([i[1] for i in candidates])[0][0], mode([i[2] for i in candidates])[0][0])
            cv2.circle(img,(ball[0],ball[1]),ball[2],(0,255,0),2)
        cnt+=1
        if ball is not None:
            angle,intersection = face_angle(ball,closed,top_point,left_point,right_point,line_center)
            

        
        cv2.imshow('image',img)
        if angle != None:
            ratio = club_path_ratio(top_point,left_point,right_point, intersection,closed)
            club_path_point=club_path_points(deq,ratio,closed)
            club_path(club_path_point,intersection,line_center)
            cv2.line(img,club_path_point,intersection,(0,0,255),2)
            

            cv2.imwrite('results.jpg', img)
            
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            

        if cv2.waitKey(1) & 0xFF == ord('q'):   
                break

        # cv2.imwrite('results.jpg', img)
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()