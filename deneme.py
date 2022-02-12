import cv2


cap=cv2.VideoCapture("rtsp://admin:Password@192.168.1.64/Streaming/channels/0001/transportmode=unicast")

while True:
    _,frame=cap.read()
    cv2.imshow("frame",frame)
    cv2.waitKey(1)
    #MHJPG67437#