import cv2
import numpy as np

import utils

def main():
  capture = cv2.VideoCapture(0)
  target_image = cv2.imread('resources/book_cover.jpg')
  output_image = cv2.imread('resources/dog.jpeg')
  
  # detector
  orb_detector = cv2.ORB_create(nfeatures=1000)
  
  # features e descriptors da imagem alvo
  image_keypoints, image_descriptor= orb_detector.detectAndCompute(target_image, None)
  
  while True:
    # obtendo a imagem da webcam
    _, webcam_image = capture.read()
    
    # features e descriptors da imagem da webcam
    webcam_keypoints, webcam_descriptor= orb_detector.detectAndCompute(webcam_image, None)
    
    # encontrando os matches entre as duas imagens
    good_matches = utils.get_good_matches(image_descriptor, webcam_descriptor)
    
    # comparando os keypoints e os bons matches
    image_features = cv2.drawMatches(target_image, image_keypoints, webcam_image, webcam_keypoints, good_matches, None, flags=2)
    
    # só prossegue se há mais de 20 bons matches
    if len(good_matches) > 20:
      matrix, mask = utils.get_homography(image_keypoints, webcam_keypoints, good_matches)
      
      # bounding box - contorno da imagem da webcam
      bounding_box = utils.get_bounding_box(target_image, webcam_image, matrix)
      
      # transformar a imagem de saída
      warped_image = utils.transform_output(target_image, webcam_image, output_image, matrix)
    
      utils.display(target_image, webcam_image, image_features, bounding_box, warped_image)
  
main()