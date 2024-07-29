import cv2
import numpy as np

def main(img):
  #QUESTION 9
  # equ = cv2.equalizeHist(img) 
  # cv2.imshow('image', equ)
  # cv2.waitKey(0)
  # img = equ
  
  #QUESTION 8
  # kernel = np.ones((5,5),np.float32)/25
  # dst = cv2.filter2D(img,-1,kernel)
  # img = dst
  # cv2.imshow('image', dst)
  # cv2.waitKey(0)
  
  #compute point projection
  f = 35
  ku = 10 #pixels/mm
  kv = 10 #pixels/mm
  au = 250 #pixels
  av = 135 #pixels
  cX = 0
  cY = 130
  cZ = 500
  
  point_coordiantes = np.array([cX/cZ, cY/cZ ,1])
  pixel_coordinates = get_pixel_coordinates(point_coordiantes, f, ku, kv, au, av)

#QUESTION 10
  # road_pixel_values = average_pixel_around_value(img, pixel_coordinates)
#ELSE
  # road_pixel_values = img[int(pixel_coordinates[0])][int(pixel_coordinates[1])]

  
  print (road_pixel_values)
  binary_img = binarize_img(img, road_pixel_values)
  filtered_img = image_filter(binary_img)
  selected_img = select_area(filtered_img, pixel_coordinates)
  
  
  # cv2.circle(img, (int(pixel_coordinates[0]), int(pixel_coordinates[1])), 7, (255,255,255), -1)
  # cv2.imshow('image', img)
  # cv2.waitKey(0)
  
#QUESTION 1
def get_pixel_coordinates(points_coordinates, f, ku, kv, au, av):
  k = np.array([[ku*f, 0, au], [0, kv*f, av], [0, 0, 1]])
  pixel_coordinates = k@points_coordinates
  return pixel_coordinates

#QUESTION 3

def binarize_img(img, threshold):
  binary = np.zeros((img.shape[0], img.shape[1]), np.uint8)
  binary = cv2.threshold(img, threshold-10, 255, cv2.THRESH_BINARY)
  remove_sky = cv2.threshold(img, threshold+60, 255, cv2.THRESH_BINARY)
  binary = binary[1] - remove_sky[1]
  cv2.imshow('image_binary', binary)
  cv2.waitKey(0)
  return binary

#QUESTION 4
def image_filter(img):
  structuring_element_1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
  # kernel = np.ones((5,5),np.float32)/25
  filtered = cv2.dilate(img,structuring_element_1,iterations = 3)
  filtered = cv2.erode(filtered,structuring_element_1,iterations = 10)
  cv2.imshow('image_binary', filtered)
  cv2.waitKey(0)
  return filtered


#QUESTION 5
def select_area(img, pixel_coordinates):
  height, width = img.shape
  elements_to_check = []
  elements_to_check.append([int(pixel_coordinates[0]), int(pixel_coordinates[1])])
  new_image = np.zeros(img.shape, np.uint8)
  while elements_to_check:
    element = elements_to_check.pop(0)
    if element[0] >= 0 and element[0] < height and element[1] >= 0 and element[1] < width:
      if img[element[0], element[1]] == 255:
        if new_image[element[0], element[1]] != 255:
          new_image[element[0], element[1]] = 255
          elements_to_check.append([element[0]-1, element[1]])
          elements_to_check.append([element[0]+1, element[1]])
          elements_to_check.append([element[0], element[1]-1])
          elements_to_check.append([element[0], element[1]+1])
      else:
        new_image[element[0], element[1]] = 0
    
  cv2.imshow('image_binary', new_image)
  cv2.waitKey(0)
  return new_image

def average_pixel_around_value(img, pixel_coordiantes):
  kernel = np.ones((5,5),np.float32)/25
  dst = cv2.filter2D(img,-1,kernel)
  return dst[int(pixel_coordiantes[0])][int(pixel_coordiantes[1])]
  
if __name__ == '__main__':
  img = cv2.imread('ressources/road.png', 0)
  img3a = cv2.imread('ressources/road_noise1.png', 0)
  img3b = cv2.imread('ressources/road_noise2.png', 0)
  img3c = cv2.imread('ressources/road_noise3.png', 0)
  # main(img)
  # main(img3a)
  # main(img3b)
  main(img3c)