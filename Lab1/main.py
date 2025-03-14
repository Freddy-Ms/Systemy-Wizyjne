import cv2


image = cv2.imread("image.jpg")
cv2.circle(image,(100,100),50,(0,0,255),-1)
cv2.rectangle(image,(200,200),(300,300),(0,255,0),-1)
cv2.line(image,(400,400),(500,500),(255,0,0),5)
cv2.ellipse(image,(300,500),(100,50),0,0,360,(0,255,255),-1)
cv2.imshow("image",image)
cv2.imwrite("processedImage.jpg",image)
cv2.waitKey(0)