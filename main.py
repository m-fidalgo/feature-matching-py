import cv2

def main():
  capture = cv2.VideoCapture(0)
  target_image = cv2.imread('resources/book_cover.jpg')
  
  # detector
  orb_detector = cv2.ORB_create(nfeatures=1000)
  
  # features e descriptors da imagem alvo
  image_keypoints, image_descriptor= orb_detector.detectAndCompute(target_image, None)
  
  # desenhando os keypoints na imagem
  target_image = cv2.drawKeypoints(target_image, image_keypoints, None)
  
  while True:
    success, webcam_image = capture.read()
    
    # features e descriptors da imagem da webcam
    webcam_keypoints, webcam_descriptor= orb_detector.detectAndCompute(webcam_image, None)
  
    # desenhando os keypoints na imagem da webcam
    webcam_image = cv2.drawKeypoints(webcam_image, webcam_keypoints, None)
    
    cv2.imshow("webcam", webcam_image)
    cv2.imshow("target", target_image)
    cv2.waitKey(1)
  
main()