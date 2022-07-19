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

def get_destinations(target_image, matrix):
  image_height, image_width, _ = target_image.shape
  
  # pontos baseados no tamanho da imagem alvo
  sources = np.float32([[0,0], [0,image_height], [image_width, image_height], [image_width, 0]]).reshape(-1, 1, 2)
  
  # pontos de destino a partir da matriz
  destinations = cv2.perspectiveTransform(sources, matrix)
  
  return destinations

def get_bounding_box(webcam_image, destinations):
  # gerar o contorno do bounding box
  return cv2.polylines(webcam_image.copy(), [np.int32(destinations)], True, (255,0,255), 3)

def transform_output(target_image, webcam_image, output_image, matrix):
  # tamanhos
  target_height, target_width, _ = target_image.shape
  webcam_height, webcam_width, _ = webcam_image.shape
  
  # assegurar que as imagens têm o mesmo tamanho
  output_image = cv2.resize(output_image, (target_width, target_height))
  
  # transformar a imagem de saída
  return cv2.warpPerspective(output_image, matrix, (webcam_width, webcam_height))

def get_masked_image(webcam_image, destinations):
  webcam_height, webcam_width, _ = webcam_image.shape
  
  # máscara preta cobrindo toda a imagem da webcam
  inverse_mask = np.zeros((webcam_height, webcam_width), np.uint8)
  
  # preenchendo os pontos de destino com branco
  cv2.fillPoly(inverse_mask, [np.int32(destinations)], (255, 255, 255))
  
  # invertendo as cores
  mask = cv2.bitwise_not(inverse_mask)
  
  # preencher os locais que não correspondem à imagem alvo com a imagem da webcam
  masked_image = cv2.bitwise_and(webcam_image, webcam_image, mask = mask)
  return masked_image

def overlay_images(masked_image, output_image):
  return cv2.bitwise_or(output_image, masked_image)
      
def stackImages(images,scale,lables=[]):
    height, width = 400, 300
    rows, cols = len(images), len(images[0])
    
    rows_available = isinstance(images[0], list)
    if rows_available:
        for x in range ( 0, rows):
            for y in range(0, cols):
                images[x][y] = cv2.resize(images[x][y], (width,height), None, scale, scale)
                if len(images[x][y].shape) == 2: images[x][y]= cv2.cvtColor( images[x][y], cv2.COLOR_GRAY2BGR)
        
        blank_image = np.zeros((height, width, 3), np.uint8)
        horizontal = [blank_image]*rows
        horizontal_concat = [blank_image]*rows
        
        for x in range(0, rows):
            horizontal[x] = np.hstack(images[x])
            horizontal_concat[x] = np.concatenate(images[x])
        vertical = np.vstack(horizontal)
    else:
        for x in range(0, rows):
            images[x] = cv2.resize(images[x], (width, height), None, scale, scale)
            if len(images[x].shape) == 2: images[x] = cv2.cvtColor(images[x], cv2.COLOR_GRAY2BGR)
        horizontal = np.hstack(images)
        horizontal_concat = np.concatenate(images)
        vertical = horizontal
    return vertical