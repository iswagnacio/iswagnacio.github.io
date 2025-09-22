import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import skimage as sk
import skimage.io as skio
import time

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])

def read_image(image_path):
    image = plt.imread(image_path)
    if len(image.shape) == 3:
        image = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]

    if image.max() > 1.0:
        image = image.astype(np.float64) / 255.0

    return image

def convolution_four_loops(image, kernel):

    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    output = np.zeros_like(image)
    
    for i in range(img_h):
        for j in range(img_w):
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    output[i, j] += padded_image[i + ki, j + kj] * kernel[ki, kj]
    
    return output

def convolution_two_loops(image, kernel):

    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    output = np.zeros_like(image)
    
    for i in range(img_h):
        for j in range(img_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)
    
    return output

def create_box_filter(size):
    return np.ones((size, size)) / (size * size)

def finite_difference(image_path):
    
    image = read_image(image_path)

    dx = convolve2d(image, Dx, mode='same', boundary='fill')
    dy = convolve2d(image, Dy, mode='same', boundary='fill')

    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    threshold = np.percentile(gradient_magnitude, 78)
    edge_image = gradient_magnitude > threshold

    plt.imshow(edge_image, cmap='gray')
    plt.axis('off')
    plt.show()

    return dx, dy, gradient_magnitude, edge_image, threshold

def gaussian(image_path, sigma=1.0, kernel_size=15):
    image = read_image(image_path)
    
    gaussian_1d = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_2d = gaussian_1d @ gaussian_1d.T
    blurred_image = convolve2d(image, gaussian_2d, mode='same', boundary='fill')

    dx_method1 = convolve2d(blurred_image, Dx, mode='same', boundary='fill')
    dy_method1 = convolve2d(blurred_image, Dy, mode='same', boundary='fill')
    gradient_mag_method1 = np.sqrt(dx_method1**2 + dy_method1**2)
    
    threshold1 = np.percentile(gradient_mag_method1, 85)
    edge_method1 = gradient_mag_method1 > threshold1

    dog_x = convolve2d(gaussian_2d, Dx, mode='same')
    dog_y = convolve2d(gaussian_2d, Dy, mode='same')
    dx_method2 = convolve2d(image, dog_x, mode='same', boundary='fill')
    dy_method2 = convolve2d(image, dog_y, mode='same', boundary='fill')
    gradient_mag_method2 = np.sqrt(dx_method2**2 + dy_method2**2)
    
    threshold2 = np.percentile(gradient_mag_method2, 85)
    edge_method2 = gradient_mag_method2 > threshold2

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Original, Blurred, and DoG filters
    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(blurred_image, cmap='gray')
    axes[0,1].set_title(f'Gaussian Blurred (σ={sigma})')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(dog_x, cmap='gray')
    axes[0,2].set_title('DoG X Filter')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(dog_y, cmap='gray')
    axes[0,3].set_title('DoG Y Filter')
    axes[0,3].axis('off')
    
    # Row 2: Method 1 results (blur first)
    axes[1,0].imshow(dx_method1, cmap='gray')
    axes[1,0].set_title('dx (Method 1: Blur first)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(dy_method1, cmap='gray')
    axes[1,1].set_title('dy (Method 1: Blur first)')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(gradient_mag_method1, cmap='gray')
    axes[1,2].set_title('Gradient Mag (Method 1)')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(edge_method1, cmap='gray')
    axes[1,3].set_title(f'Edges (Method 1, t={threshold1:.3f})')
    axes[1,3].axis('off')
    
    # Row 3: Method 2 results (DoG filters)
    axes[2,0].imshow(dx_method2, cmap='gray')
    axes[2,0].set_title('dx (Method 2: DoG)')
    axes[2,0].axis('off')
    
    axes[2,1].imshow(dy_method2, cmap='gray')
    axes[2,1].set_title('dy (Method 2: DoG)')
    axes[2,1].axis('off')
    
    axes[2,2].imshow(gradient_mag_method2, cmap='gray')
    axes[2,2].set_title('Gradient Mag (Method 2)')
    axes[2,2].axis('off')
    
    axes[2,3].imshow(edge_method2, cmap='gray')
    axes[2,3].set_title(f'Edges (Method 2, t={threshold2:.3f})')
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_convolutions():
    selfie_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/cameraman.png'

    selfie = read_image(selfie_path)

    box_filter = create_box_filter(9)  

    result_4loop_box = convolution_four_loops(selfie, box_filter)
    result_2loop_box = convolution_two_loops(selfie, box_filter)
    result_scipy_box = convolve2d(selfie, box_filter, mode='same', boundary='fill')

    result_scipy_dx = convolve2d(selfie, Dx, mode='same', boundary='fill')
    result_scipy_dy = convolve2d(selfie, Dy, mode='same', boundary='fill')
  

if __name__ == "__main__":
    #test_convolutions()
    #finite_difference('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/cameraman.png')
    gaussian('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/cameraman.png', sigma=1.7, kernel_size=15)