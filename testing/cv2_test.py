import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, frame = cap.read()
if not ret:
    print("Cannot get frame")
    exit()

cv2.imshow("frame", frame)
if cv2.waitKey(0) == ord('q'):
    pass

cap.release()
cv2.destroyAllWindows()