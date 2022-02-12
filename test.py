import cv2

cap = cv2.VideoCapture('omer-emre.mp4')

frame_no = 0
while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        print("for frame : " + str(frame_no) + "   timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))
    else:
        break
    frame_no += 1

cap.release()