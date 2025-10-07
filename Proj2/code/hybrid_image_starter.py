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
    
    return gaussian_pyramid, laplacian_pyramid

def frequency_analysis(im1, im2, hybrid, sigma1, sigma2, output_dir):
    im1_gray = np.mean(im1, axis=2)
    im2_gray = np.mean(im2, axis=2) 
    hybrid_gray = np.mean(hybrid, axis=2)
    low_freq_im1 = gaussian_filter(im1_gray, sigma=sigma1)
    high_freq = im1_gray - low_freq_im1
    low_freq = gaussian_filter(im2_gray, sigma=sigma2)
    
    images = [im1_gray, im2_gray, high_freq + 0.5, low_freq, hybrid_gray]
    titles = ['Input Image 1', 'Input Image 2', 'High-pass Filtered', 'Low-pass Filtered', 'Hybrid Image']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[0,i].imshow(img, cmap='gray')
        axes[0,i].set_title(title)
        axes[0,i].axis('off')
        fft_result = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))
        axes[1,i].imshow(fft_result, cmap='hot')
        axes[1,i].set_title(f'FFT Magnitude - {title}')
        axes[1,i].axis('off')
    
    plt.tight_layout()
    plt.show()

# First load images

# high sf
im1 = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/DerekPicture.jpg')/255

# low sf
im2 = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/nutmeg.jpg')/255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im2, im1)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = 10.0   # For high-pass filtering (smaller = keeps more high freq)
sigma2 = 8.0  # For low-pass filtering (larger = keeps more low freq)
output_dir = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media'

hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
#frequency_analysis(im1_aligned, im2_aligned, hybrid, sigma1, sigma2, output_dir=output_dir)

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(im1_aligned, cmap='gray' if len(im1_aligned.shape) == 2 else None)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(im2_aligned, cmap='gray' if len(im2_aligned.shape) == 2 else None)  
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(hybrid, cmap='gray')
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