import cv2
import pandas as pd

# Load the image
img = cv2.imread("Cat_August_2010-4.jpg")

# Load the color data
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv("colors.csv", names=index, header=None)

# Create the window + variables
cv2.namedWindow("Color Detection Window")
clicked = False
r = g = b = xpos = ypos = 0

# Callback Function
def call_back_function(event, x, y, flags, param):
    """Calculates the RGB values of the pixel that was double-clicked and the coordinates are saved."""
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos,clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)

# Calculate distance to get color name
def get_color_name(B, G, R):
    minimum = 10000
    # d = abs(Red — ithRedColor) + (Green — ithGreenColor) + (Blue — ithBlueColor)
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

cv2.setMouseCallback("Color Detection Window", call_back_function)

# Draw Function
while(True):
    cv2.imshow("Color Detection Window", img)
    if (clicked):
        cv2.rectangle(img, (20,20), (750, 60), (b, g, r), -1)
        text = f"{get_color_name(r,g,b)}"
        cv2.putText(img, text, (50,50), 2, 0.8, (0,0,0), 2, cv2.LINE_AA)
        if (r + g + b <= 500):
            cv2.putText(img, text, (50,50), 2, 0.8, (255,255,255), 2, cv2.LINE_AA)
        clicked = False

        # End when the user presses the "ESC" key
        if cv2.waitKey(20) & 0xFF == 27:
            break

cv2.destroyAllWindows()


# a = "asdf"
# b = 45
# c = 3

# "Hello: asdf 48"
# print("Hello: " + a + str(b + c))
# print(f"Hello: {a} {b + c}")










# cv2.imshow("Cat Image", img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()