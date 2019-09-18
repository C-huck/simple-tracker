import cv2
import numpy as np
#from matplotlib import pyplot as plt 

def sum_distance(points):
    """
    Calculate point-to-point distances and
    sums them. Grand total is total distance
    selected object travels, in pixels
    """
    dist = 0
    for i in range(1,len(points)-1):
        x = (points[i][0] - points[i+1][0])**2
        y = (points[i][1] - points[i+1][1])**2
        d = (x+y)**(0.5)
        dist+=d
    return dist 

def opt_flow(cap,frame,p0,dim1,dim2):
    """
    Compute frame-by-frame optical flow of pixels selected in ROI
    Displays video and saves it with running displacement
    """
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (dim1,dim2), #dim1 and dim2 are inhereted from ROI box
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    mask = np.zeros_like(frame)
    color = np.random.randint(0,255,(100,3))
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while(True):
        ret,frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        points.append(list(good_new)[0])
    
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 5)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        
        #writing and displaying video
        distance = round(sum_distance(points),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,str(distance),(10,frame_height-100), font, 4,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow('frame',img)
        out.write(img)

        #Press ESC to quit early
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
        #Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    return points
    

if __name__ == '__main__' :
 
    # Read first frame
    g = str(input("Video name + ext : "))
    cap = cv2.VideoCapture(g)
    ret,frame = cap.read()

    # Select ROI
    r = cv2.selectROI(frame)
    x_bar = round(r[0]+r[2]/2.0)
    y_bar = round(r[1]+r[3]/2.0)
    center = (int(x_bar),int(y_bar))
    p0 = np.float32(np.asarray([[[x_bar,y_bar]]]))
    cv2.destroyWindow('ROI selector')
    
    #call optical flow
    points = opt_flow(cap,frame,p0,r[2],r[3])
    
    #print distance of tracked object in console
    print(sum_distance(points))
    
    #hold until exit key pressed ('q')
    cv2.waitKey(0)
    
    
