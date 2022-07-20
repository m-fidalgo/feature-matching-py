import cv2
import object_loader
import utils

def mode_2d(target_image, webcam_image, image_features, bounding_box, masked_image, matrix):
  # imagem que será sobreposta
  output_image = cv2.imread('src/resources/book_cover_new.jpg')
  
  # transformar a imagem de saída
  output_image = utils.transform_output(target_image, webcam_image, output_image, matrix)      

  result = utils.overlay_images(masked_image, output_image)

  # exibindo as imagens
  stacked_images = utils.stackImages(([target_image, webcam_image, bounding_box], [image_features, output_image, result]), 0.5)
  cv2.imshow("images", stacked_images)
  cv2.waitKey(1)

def mode_3d(target_image, bounding_box, matrix):
  # carregar o objeto
  object = object_loader.object("src/resources/chair.obj", swapyz=True)
  
  # matriz de projeção 3d
  projection = utils.get_projection_matrix(matrix)
  
  # projeção do modelo
  frame = utils.render(bounding_box.copy(), object, projection, target_image, 8)
  cv2.imshow("frame", frame)
  cv2.waitKey(1)

def main(mode):
  capture = cv2.VideoCapture(0)
  target_image = cv2.imread('src/resources/book_cover.jpg')
  
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
      
      # máscara sobre a imagem da webcam
      masked_image = utils.get_masked_image(webcam_image, destinations)
      
      if mode == "2d":
        mode_2d(target_image, webcam_image, image_features, bounding_box, masked_image, matrix)
      elif mode == "3d":
        mode_3d(target_image, bounding_box, matrix)
      
main("3d")