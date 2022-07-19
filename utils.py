import cv2
import numpy as np

def get_good_matches(image_descriptor, webcam_descriptor):
  matcher = cv2.BFMatcher()
  matches = matcher.knnMatch(image_descriptor, webcam_descriptor, k=2)
  
  good_matches = []
  
  for m, n in matches:
    # se a distância é suficientemente pequena, é considerado um bom match
    if m.distance < 0.75 * n.distance:
      good_matches.append(m)
  
  return good_matches

def get_homography(image_keypoints, webcam_keypoints, good_matches):
  source_points = np.float32([image_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
  destination_points = np.float32([webcam_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
  
  # matriz de homografia - relaciona as duas imagens
  matrix, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5)
  
  print(matrix)
  return matrix, mask

def get_bounding_box(target_image, webcam_image, matrix):
  image_height, image_width, _ = target_image.shape
  
  # pontos baseados no tamanho da imagem alvo
  points = np.float32([[0,0], [0,image_height], [image_width, image_height], [image_width, 0]]).reshape(-1, 1, 2)
  
  # pontos de destino a partir da matriz
  destinations = cv2.perspectiveTransform(points, matrix)
  
  # gerar o contorno do bounding box
  return cv2.polylines(webcam_image, [np.int32(destinations)], True, (255,0,255), 3)

def transform_output(target_image, webcam_image, output_image, matrix):
  # tamanhos
  target_height, target_width, _ = target_image.shape
  webcam_height, webcam_width, _ = webcam_image.shape
  
  # assegurar que as imagens têm o mesmo tamanho
  output_image = cv2.resize(output_image, (target_width, target_height))
  
  # transformar a imagem de saída
  return cv2.warpPerspective(output_image, matrix, (webcam_width, webcam_height))

def display(target_image, webcam_image, image_features, bounding_box, output_image):
  cv2.imshow("features", image_features)
  #cv2.imshow("target", target_image)
  #cv2.imshow("webcam", webcam_image)
  cv2.imshow("bounding box", bounding_box)
  cv2.imshow("output", output_image)
  cv2.waitKey(0)