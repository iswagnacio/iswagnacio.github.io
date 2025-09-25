import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from skimage import io, img_as_float
import os

def create_gaussian_kernel(size, sigma):
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def unsharp_masking(image, sigma=0.6, alpha=1.2):
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    sharpened = np.zeros_like(image)
    blurred = np.zeros_like(image)
    high_freq = np.zeros_like(image)

    for channel in range(image.shape[2]):
        blurred[:, :, channel] = convolve2d(image[:, :, channel], gaussian_kernel, mode='same', boundary='symm')
        high_freq[:, :, channel] = image[:, :, channel] - blurred[:, :, channel]
        sharpened[:, :, channel] = image[:, :, channel] + alpha * high_freq[:, :, channel]

    sharpened = np.clip(sharpened, 0, 1)
    
    return sharpened, blurred, high_freq

def create_unsharp_kernel(sigma=0.6, alpha=1.2):
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    identity_kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    identity_kernel[center, center] = 1.0

    unsharp_kernel = (1 + alpha) * identity_kernel - alpha * gaussian_kernel
    
    return unsharp_kernel

def unsharp_masking_single_conv(image, sigma=0.6, alpha=1.2):
    unsharp_kernel = create_unsharp_kernel(sigma, alpha)    
    sharpened = np.zeros_like(image)

    for channel in range(image.shape[2]):
        sharpened[:, :, channel] = convolve2d(image[:, :, channel], unsharp_kernel, 
                                            mode='same', boundary='symm')
    
    sharpened = np.clip(sharpened, 0, 1)
    return sharpened

def blur_and_resharpen_test(image, blur_sigma=2.0, sharp_sigma=1.0, alpha=1.2):
    blur_kernel_size = int(6 * blur_sigma) + 1
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    
    blur_kernel = create_gaussian_kernel(blur_kernel_size, blur_sigma)
    blurred_image = np.zeros_like(image) 

    for channel in range(image.shape[2]):
        blurred_image[:, :, channel] = convolve2d(image[:, :, channel], blur_kernel, 
                                                mode='same', boundary='symm')
    resharpened, _, _ = unsharp_masking(blurred_image,sigma=sharp_sigma, alpha=alpha)
    
    return blurred_image, resharpened

def visualize_high_frequencies(high_freq):
    hf_vis = high_freq - np.min(high_freq)
    hf_vis = hf_vis / np.max(hf_vis) if np.max(hf_vis) > 0 else hf_vis
    return hf_vis

def process_image_sharpening(image_path, output_dir='/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2'):
    os.makedirs(output_dir, exist_ok=True)
    image = io.imread(image_path)
    image = img_as_float(image)
    
    sigma = 0.6
    alpha_values = [0.5, 1.0, 1.5, 2.0]

    _, blurred, high_freq = unsharp_masking(image, sigma=sigma, alpha=alpha_values[0])
    high_freq_vis = visualize_high_frequencies(high_freq)
    sharpened_results = {}
    
    for alpha in alpha_values:
        sharpened, _, _ = unsharp_masking(image, sigma=sigma, alpha=alpha)

        filename = f'sharpened_alpha_{alpha:.1f}.png'
        plt.imsave(f'{output_dir}/{filename}', sharpened)
        sharpened_results[alpha] = sharpened

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred)
    axes[0, 1].set_title(f'Blurred (σ={sigma})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(high_freq_vis)
    axes[0, 2].set_title('High Frequencies')
    axes[0, 2].axis('off')

    for i, alpha in enumerate(alpha_values[:3]):
        axes[1, i].imshow(sharpened_results[alpha])
        axes[1, i].set_title(f'Sharpened (α={alpha})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sharpening_process_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, alpha in enumerate(alpha_values):
        axes[i].imshow(sharpened_results[alpha])
        axes[i].set_title(f'α = {alpha}')
        axes[i].axis('off')
    
    plt.suptitle(f'Sharpening Amount Comparison (σ={sigma})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/alpha_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'original': image,
        'blurred': blurred,
        'high_freq': high_freq,
        'sharpened_results': sharpened_results,
        'alpha_values': alpha_values
    }

if __name__ == "__main__":
    im_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/_DSF0487.jpg'

    try:
        results = process_image_sharpening(im_path)
    except FileNotFoundError:
        print(f"Could not find {im_path}")