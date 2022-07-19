import cv2

def get_good_matches(image_descriptor, webcam_descriptor):
  matcher = cv2.BFMatcher()
  matches = matcher.knnMatch(image_descriptor, webcam_descriptor, k=2)
  
  good_matches = []
  
  for m, n in matches:
    # se a distância é suficientemente pequena, é considerado um bom match
    if m.distance < 0.75 * n.distance:
      good_matches.append(m)
  
  print(len(good_matches))
  return good_matches

def display(target_image, webcam_image, image_features):
  cv2.imshow("features", image_features)
  cv2.imshow("target", target_image)
  cv2.imshow("webcam", webcam_image)
  cv2.waitKey(1)