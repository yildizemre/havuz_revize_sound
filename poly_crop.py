import cv2
import time

# Get the camera 0 for the camera index
cap = cv2.VideoCapture("/home/emreyildiz/havuz_guncel/yolov4-deepsort/data/video/test.mp4")

# Get camera frame rate
fps = cap.get(cv2.CAP_PROP_FPS)

# Set camera resolution
cap.set(3, 840)
cap.set(4, 480)
# At this time, the reset resolution is obtained
frame_rate = 10
prev = 0

while True:
    start = time.time()
    time.sleep(5)
    time_elapsed = time.time() - prev
    res, image = cap.read()
    print(time_elapsed)
    print("*****")
    print(1./frame_rate)
    if time_elapsed > 1./frame_rate:
        prev = time.time()

        # Do something with your image here.
        ret, frame = cap.read()

    
    # If each frame is read correctly, ret returns True and frame returns the data of the current frame
    
    # Save video
    

    # Show camera stream
        cv2.imshow("frame", frame)

        # Press q to exit the display
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        end = time.time()
        print(f"Read a frame time: {(end-start)*500:.3f}ms")
    # The unit of sleep is seconds
    # time.sleep(1/100)

cap.release()   # Release the camera

cv2.destroyAllWindows()  # Close all image windows