from analyzer import Analyzer

test_video = "videos/Fruits.mp4"

if __name__ == "__main__":
    analysis_engine = Analyzer()
    analysis_engine.start_video_capture(capture_path=test_video)


# import cv2
# import pandas as pd

# # # Load the image
# img = cv2.imread("Cat_August_2010-4.jpg")

# width = 500
# height = 300

# down_size = (width, height)
# img = cv2.resize(img, down_size, interpolation=cv2.INTER_LINEAR)

# # # scale_x = 0.5
# # # scale_y = 0.5

# # # scale = 0.75
# # # resized_img1 = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
# # # resized_img2 = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

# # """
# #     INTER_AREA: INTER_AREA uses pixel area relation for resampling. This is best suited for reducing the size of an image (shrinking). When used for zooming into the image, it uses the INTER_NEAREST method.
# #     INTER_CUBIC: This uses bicubic interpolation for resizing the image. While resizing and interpolating new pixels, this method acts on the 4×4 neighboring pixels of the image. It then takes the weights average of the 16 pixels to create the new interpolated pixel.
# #     INTER_LINEAR: This method is somewhat similar to the INTER_CUBIC interpolation. But unlike INTER_CUBIC, this uses 2×2 neighboring pixels to get the weighted average for the interpolated pixel.
# #     INTER_NEAREST: The INTER_NEAREST method uses the nearest neighbor concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.
# # """

# # font = cv2.FONT_HERSHEY_SIMPLEX
# # org = (50, 50)
# # fontScale = 1
# # color = (0, 0, 0) # BGR
# # thickness = 2

# # resized_img = cv2.putText(resized_img, "Fluffy", org, font, fontScale, color, thickness, cv2.LINE_AA)


# # Load the color data
# index = ["color", "color_name", "hex", "R", "G", "B"]
# csv = pd.read_csv("Simplified_colors.csv", names=index, header=None)

# # Create the window + variables
# cv2.namedWindow("Color Detection Window")
# clicked = False
# r = g = b = xpos = ypos = 0

# # Callback Function
# def call_back_function(event, x, y, flags, param):
#     """Calculates the RGB values of the pixel that was double-clicked and the coordinates are saved."""
#     # print(f"User has clicked! Here is the event: {event}, we need {cv2.EVENT_LBUTTONDBLCLK}")
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         global b,g,r,xpos,ypos,clicked
#         clicked = True
#         xpos = x
#         ypos = y
#         b,g,r = img[y,x]
#         b = int(b)
#         g = int(g)
#         r = int(r)

# # Calculate distance to get color name
# def get_color_name(B, G, R):
#     minimum = 10000
#     # d = abs(Red — ithRedColor) + (Green — ithGreenColor) + (Blue — ithBlueColor)
#     for i in range(len(csv)):
#         d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
#         if (d <= minimum):
#             minimum = d
#             cname = csv.loc[i, "color_name"]
#     return cname

# cv2.setMouseCallback("Color Detection Window", call_back_function)

# # Draw Function
# while(True):
#     cv2.imshow("Color Detection Window", img)
#     if (clicked):
#         cv2.rectangle(img, (20,20), (750, 60), (b, g, r), -1)
#         text = f"{get_color_name(b,g,r)} R:{r} G:{g} B:{b}"
#         cv2.putText(img, text, (50,50), 2, 0.8, (0,0,0), 2, cv2.LINE_AA)
#         if (r + g + b <= 500):
#             cv2.putText(img, text, (50,50), 2, 0.8, (255,255,255), 2, cv2.LINE_AA)
#         clicked = False

#     # End when the user presses the "ESC" key
#     if cv2.waitKey(20) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()

# # cv2.imshow("Cat Image", resized_img)

# # cv2.waitKey(0)

# cv2.destroyAllWindows()