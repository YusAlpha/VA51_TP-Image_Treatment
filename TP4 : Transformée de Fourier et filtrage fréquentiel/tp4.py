import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal

def main():
  img1 = cv2.imread("ressources/road1.png", 0)
  cv2.imshow("img1", img1)

  fourier = np.fft.fft2(img1)
  shifted = np.fft.fftshift(fourier)
  magnitude = np.abs(shifted)
  log_magnitude = 10 * np.log10(magnitude)
  
  # masked_center_fourier = mask_center(shifted)
  masked_border_fourier = mask_border(shifted)
  
  
  # filtered = np.fft.ifftshift(masked_center_fourier)
  filtered2 = np.fft.ifftshift(masked_border_fourier)
  
  # filtered_image = np.fft.ifft2(filtered).real
  filtered_image2 = np.fft.ifft2(filtered2).real
  
  # plt.imshow(filtered_image, cmap="gray")
  # plt.show()
  plt.imshow(filtered_image2, cmap="gray")
  plt.show()
  # plt.imshow(masked_fourier)
  # plt.show()
  
def mask_center(fourier):
  fourier[325:375, 325:375] = 0
  return fourier

def mask_border(fourier):
  fourier[0:325] = 0
  fourier[: , 375:699] = 0
  fourier[375:699] = 0
  fourier[:, 0:325] = 0
  return fourier


def main2():
  img2 = cv2.imread("ressources/road2.png", 0)
  fourier = np.fft.fft2(img2)
  shifted = np.fft.fftshift(fourier)
  # magnitude = np.abs(shifted)
  # log_magnitude = 10 * np.log10(magnitude)
  
  gaussian_filter = get_gaussian_filter(shifted)
  
  plt.imshow(gaussian_filter, cmap="gray")
  plt.show()
  
  filtered_fourier = shifted * gaussian_filter
  # filtered_fourier2 = np.convolve2d(shifted, gaussian_filter, mode="same")
  filtered_fourier2 = signal.convolve2d(shifted, gaussian_filter, mode="same")
  print(filtered_fourier2)
  exit()
  
  filtered = np.fft.ifftshift(filtered_fourier)
  filtered2 = np.fft.ifftshift(filtered_fourier2)
  filtered_image = np.fft.ifft2(filtered)
  filtered_image2 = np.fft.ifft2(filtered2)
  filtered_image = np.real(filtered_image)
  filtered_image2 = np.real(filtered_image2)
  
  plt.imshow(filtered_image2, cmap="gray")
  plt.show()
  # plt.figure()
  # f, axarr = plt.subplots(2,1) 
    
  # axarr[0].imshow(filtered_image, cmap="gray")
  # axarr[1].imshow(filtered_image2, cmap="gray")
  # plt.show()
  
  
def get_gaussian_filter(img):
  sigma = 10
  new_image = np.zeros(img.shape)
  height, width = img.shape
  for i in range(height):
    for j in range(width):
      new_image[i][j] = np.exp(-((i - height/2)**2 + (j - width/2)**2)/(2*sigma**2))
      new_image[i][j] = new_image[i][j]*255
  return new_image


if __name__ == "__main__":
  # main()
  main2()