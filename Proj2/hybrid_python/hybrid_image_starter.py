import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from align_image_code import align_images

def hybrid_image(im1, im2, sigma1, sigma2):
    if len(im1.shape) == 2:
        im1 = np.stack([im1, im1, im1], axis=2)
    if len(im2.shape) == 2:
        im2 = np.stack([im2, im2, im2], axis=2)
    print("Image shapes:", im1.shape, im2.shape)
    assert im1.shape == im2.shape, f"Image size mismatch: {im1.shape} vs {im2.shape}"

    hybrid = np.zeros_like(im1)
    
    for channel in range(3):  
        # Low-pass the second image
        low_freq = gaussian_filter(im2[:, :, channel], sigma=sigma2)
        
        # High-pass the first image
        low_freq_im1 = gaussian_filter(im1[:, :, channel], sigma=sigma1)
        high_freq = im1[:, :, channel] - low_freq_im1
        
        # Combine
        hybrid[:, :, channel] = high_freq + low_freq

    hybrid = np.clip(hybrid, 0, 1)
    
    return hybrid


def pyramids(image, N):
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2)
        is_color = True
    else:
        gray_image = image
        is_color = False
    
    gaussian_pyramid = []
    laplacian_pyramid = []
    current_image = gray_image.copy()
    gaussian_pyramid.append(current_image)
    
    for i in range(1, N):
        sigma = 2.0 ** i  
        blurred = gaussian_filter(gray_image, sigma=sigma)
        gaussian_pyramid.append(blurred)

    for i in range(N-1):
        laplacian = gaussian_pyramid[i] - gaussian_pyramid[i+1]
        laplacian_pyramid.append(laplacian)

    laplacian_pyramid.append(gaussian_pyramid[-1])
    

    fig, axes = plt.subplots(2, N, figsize=(15, 6))
    for i in range(N):
        axes[0, i].imshow(gaussian_pyramid[i], cmap='gray')
        axes[0, i].set_title(f'Gaussian Level {i}')
        axes[0, i].axis('off')
    
    for i in range(N):
        laplacian_display = laplacian_pyramid[i] + 0.5
        laplacian_display = np.clip(laplacian_display, 0, 1)
        axes[1, i].imshow(laplacian_display, cmap='gray')
        axes[1, i].set_title(f'Laplacian Level {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return gaussian_pyramid, laplacian_pyramid


# First load images

# high sf
im1 = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/hybrid_python/DerekPicture.jpg')/255.

# low sf
im2 = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/hybrid_python/nutmeg.jpg')/255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im2, im1)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = 8.0   # For high-pass filtering (smaller = keeps more high freq)
sigma2 = 15.0  # For low-pass filtering (larger = keeps more low freq)

hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(im1_aligned, cmap='gray' if len(im1_aligned.shape) == 2 else None)
plt.title('Derek (High Freq Source)')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(im2_aligned, cmap='gray' if len(im2_aligned.shape) == 2 else None)  
plt.title('Nutmeg (Low Freq Source)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(hybrid, cmap='gray')
plt.title('Hybrid Image')
plt.axis('off')

# Show a smaller version to simulate viewing from distance
plt.subplot(1, 4, 4)
# Downsample to simulate distance viewing
hybrid_small = hybrid[::4, ::4]  # Take every 4th pixel
plt.imshow(hybrid_small, cmap='gray')
plt.title('Hybrid (Simulated Distance)')
plt.axis('off')

plt.tight_layout()
plt.show()

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
gaussian_pyr, laplacian_pyr = pyramids(hybrid, N)