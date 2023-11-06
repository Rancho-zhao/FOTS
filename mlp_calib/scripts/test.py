import cv2

bg = cv2.imread("/home/r404/Digit_Test/digit-depth/scripts/0000.png")
img = cv2.imread("/home/r404/Digit_Test/digit-depth/images/0001.png")
ii = img - bg
cv2.imshow("ii",ii)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()