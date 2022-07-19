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
      matrix, _ = utils.get_homography(image_keypoints, webcam_keypoints, good_matches)
      
      # pontos de destino
      destinations = utils.get_destinations(target_image, matrix)
      
      # bounding box - contorno da imagem da webcam
      bounding_box = utils.get_bounding_box(webcam_image, destinations)
      
      # transformar a imagem de saída
      output_image = utils.transform_output(target_image, webcam_image, output_image, matrix)
    
      # máscara sobre a imagem da webcam
      masked_image = utils.get_masked_image(webcam_image, destinations)
    
      result = utils.overlay_images(masked_image, output_image)
    
      # exibindo as imagens
      stacked_images = utils.stackImages(([target_image, webcam_image, bounding_box], [image_features, output_image, result]), 0.5)
      cv2.imshow("images", stacked_images)
      cv2.waitKey(0)
  
main()