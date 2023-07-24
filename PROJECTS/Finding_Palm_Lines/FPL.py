import cv2
image = cv2.resize(cv2.imread(
    r"PROJECTS\Finding_Palm_Lines\hand.jpg"), (640, 640))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 150, 200, apertureSize=3)
cv2.imshow("edges in palm", edges)
cv2.waitKey(0)
edges = cv2.bitwise_not(edges)
cv2.imwrite(r"PROJECTS\Finding_Palm_Lines\palmlines.jpg", edges)
palmlines = cv2.imread(r"PROJECTS\Finding_Palm_Lines\palmlines.jpg")
img = cv2.addWeighted(palmlines, 0.2, image, 0.8, 0)
cv2.imwrite(r"PROJECTS\Finding_Palm_Lines\weighted_image.jpg", img)
