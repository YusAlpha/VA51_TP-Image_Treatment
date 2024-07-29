import numpy as np
import cv2
import os
import matplotlib.pyplot as plt 

def compute_histogram(img):
  histogram = np.zeros(256)
  for i in range(256):
    total = np.sum(img == i)
    histogram[i] = total
    
    return histogram
    
  # print(histogram)
  # plt.title("Historgram") 
  # plt.xlabel("pixel value") 
  # plt.ylabel("pixel count") 
  # plt.bar(range(256), histogram, color ='grey', width = 0.4)
  # plt.show()
    
def set_seuils(img):
  seuils = 254
  new_img = img.copy()
  new_img[new_img < seuils] = 0
  
def compute_diff(img1, img2):
  return np.abs(img1 - img2)

def recadrage_dynamique(img):
  a = 52
  b = 156
  m = 255
  for i in range(len(img)):
    img[i] = np.round(m * ((img[i] - a) / (b - a)))
    print (img[i])
    
  return img
    
    
def compute_cumulated_histogram(img):
  cumulated_histogram = np.zeros(256)
  h, w= img.shape
  for i in range(256):
    total = np.sum(img <= i)
    cumulated_histogram[i] = total/(h*w)
    
  # print(histogram)
  plt.title("Historgram") 
  plt.xlabel("pixel value")
  plt.ylabel("pixel count") 
  plt.bar(range(256), cumulated_histogram, color ='brown', width = 0.4)
  plt.show()
  
  return cumulated_histogram
  
def equalize_histogram(img, cumulated_histogram):
  new_img = img.copy()
  cumulated_histogram = cumulated_histogram
  for i in range(len(img)):
    new_img[i] = np.round((cumulated_histogram[img[i]])*255)
  return new_img

def compute_normalised_histogram(img):
  p = np.zeros(256)
  h, w = img.shape
  total_pixels = h*w
  for i in range(256):
    p[i] = float(np.sum(img == i)/total_pixels)
  return p
    
    
def compute_backgroud_weight(p):
  temp_p = p
  wb = {}
  for i in range(256):
    wb[i] = float(np.sum(temp_p[:i]))
  return wb
    
def compute_foreground_weight(p, wb):
  wf = np.zeros(256)
  for i in range(256):
    wf[i] = float(1 - wb[i])
  return wf

def compute_average_background(wb, h):
  temp_h = h
  ub = np.zeros(256)
  for i in range(256):
    temp_h = [element * index for index, element in enumerate(h)]
    if np.sum(h[:i]) != 0:
      ub[i] = np.sum(temp_h[:i])/np.sum(h[:i])
  return ub


def compute_average_foreground(wf, h):
  temp_h = h
  uf = np.zeros(256)
  for i in range(256):
    temp_h = [element * index for index, element in enumerate(h)]
    if np.sum(h[:i]) != 0:
      uf[i] = np.sum(temp_h[i:])/np.sum(h[i:])
  return uf

def compute_variance_background(wb,wf, ub, uf):
  o2b = np.zeros(256)
  for i in range(256):
    o2b[i] = wb[i]*wf[i]*pow((ub[i]-uf[i]),2)
  return o2b

if __name__ == '__main__':
  current_dir = os.getcwd()
  
  
  img1 = cv2.imread(os.path.join(current_dir, 'ressources/Ibefore.png'), cv2.IMREAD_GRAYSCALE)
  img2 = cv2.imread(os.path.join(current_dir, 'ressources/Iafter.png'), cv2.IMREAD_GRAYSCALE)
  img3 = cv2.imread(os.path.join(current_dir, 'ressources/Ihistdyn.png'), cv2.IMREAD_GRAYSCALE)
  img4 = cv2.imread(os.path.join(current_dir, 'ressources/Ihistegal.png'), cv2.IMREAD_GRAYSCALE)
  
  img_list = [img1, img2, img3, img4]
  
  
  ######### PARTIE 1 ######### 
  # diff = compute_diff(img1, img2)
  # histogram = compute_histogram(img2)
  # new_img = set_seuils(img2)

  # cv2.imshow('diff', diff)
  # cv2.waitKey(0)
  
  
  ######### PARTIE 2 ######### 
  # max = np.max(img3)
  # min = np.min(img3)
  # print("max: ", max, "min: ", min)
   
  # recadred_img = recadrage_dynamique(img3)
  # compute_histogram(recadred_img)
  # cv2.imshow('recadred_img', recadred_img)
  # cv2.imshow('image_avant_recadrage', img3)
  # cv2.waitKey(0)


  ######### PARTIE 3 ######### 
  # compute_histogram(img4)
  # print(len(img4))
  # cv2.imshow('img4', img4)
  # cumulated_histogram = compute_cumulated_histogram(img4)
  # equalized_img = equalize_histogram(img4, cumulated_histogram)
  # cv2.imshow('equalized_image', equalized_img)
  # compute_histogram(equalized_img)
  # cv2.waitKey(0) 
  
  ######### PARTIE 4 #########
  
  for img in img_list:
    h = compute_histogram(img)
    p = compute_normalised_histogram(img)
    wb = compute_backgroud_weight(p)
    wf = compute_foreground_weight(p, wb)
    ub = compute_average_background(wb, p)
    uf = compute_average_foreground(wb, p)
    
    o2b = compute_variance_background(wb, wf, ub, uf)
    
    print(np.argmax(o2b))

  
  
  
  
  
  

