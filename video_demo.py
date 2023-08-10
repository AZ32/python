import cv2

# vid_capture = cv2.VideoCapture("sample-5s.mp4")
vid_capture = cv2.VideoCapture(0)

if (vid_capture.isOpened() == False):
    print("There was an error opening the file!")
else:
    fps = vid_capture.get(5)
    print(f"Video framerate: {fps} FPS")

    frame_count = vid_capture.get(7)
    print(f"Frame count: {frame_count}")

while(vid_capture.isOpened()):
    success, frame = vid_capture.read()

    if success:
        cv2.imshow("Test Video", frame)

        key = cv2.waitKey(20)

        if key == ord('q'):
            break
    else:
        break

vid_capture.release()
cv2.destroyAllWindows()