import cv2
import utils

def main():
  capture = cv2.VideoCapture(0)
  target_image = cv2.imread('resources/book_cover.jpg')
  
  # detector
  orb_detector = cv2.ORB_create(nfeatures=1000)
  
  # features e descriptors da imagem alvo
  image_keypoints, image_descriptor= orb_detector.detectAndCompute(target_image, None)
  
  while True:
    success, webcam_image = capture.read()
    
    # features e descriptors da imagem da webcam
    webcam_keypoints, webcam_descriptor= orb_detector.detectAndCompute(webcam_image, None)
    
    # encontrando os matches entre as duas imagens
    good_matches = utils.get_good_matches(image_descriptor, webcam_descriptor)
    
    image_features = cv2.drawMatches(target_image, image_keypoints, webcam_image, webcam_keypoints, good_matches, None, flags=2)
    
    utils.display(target_image, webcam_image, image_features)
  
main()