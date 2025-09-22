import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from skimage import io, img_as_float

def create_gaussian_kernel(size, sigma):
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def unsharp_masking_color(image, sigma=1.0, alpha=0.5):
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)

    sharpened = np.zeros_like(image)
    blurred = np.zeros_like(image)
    high_freq = np.zeros_like(image)

    for channel in range(image.shape[2]):
        # Low-pass
        blurred[:, :, channel] = convolve2d(image[:, :, channel], gaussian_kernel, 
                                          mode='same', boundary='symm')
        
        # high frequencies
        high_freq[:, :, channel] = image[:, :, channel] - blurred[:, :, channel]
 
        sharpened[:, :, channel] = image[:, :, channel] + alpha * high_freq[:, :, channel]

    sharpened = np.clip(sharpened, 0, 1)
    
    return sharpened, blurred, high_freq

def create_unsharp_kernel_color(sigma=1.0, alpha=0.5):
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Gaussian kernel
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # Identity kernel
    identity_kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    identity_kernel[center, center] = 1.0

    unsharp_kernel = (1 + alpha) * identity_kernel - alpha * gaussian_kernel
    
    return unsharp_kernel

def unsharp_masking_color_single_conv(image, sigma=1.0, alpha=0.5):
    unsharp_kernel = create_unsharp_kernel_color(sigma, alpha)
    
    sharpened = np.zeros_like(image)

    for channel in range(image.shape[2]):
        sharpened[:, :, channel] = convolve2d(image[:, :, channel], unsharp_kernel, 
                                            mode='same', boundary='symm')
    
    sharpened = np.clip(sharpened, 0, 1)
    return sharpened, unsharp_kernel

def analyze_color_image_sharpness(image):
    if len(image.shape) == 3:
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        luminance = image

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = convolve2d(luminance, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(luminance, sobel_y, mode='same', boundary='symm')
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    sharpness_score = np.mean(gradient_magnitude)
    return sharpness_score

def process_color_taj():
    taj_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/taj.jpg'  # Update path as needed
    
    try:
        taj_color = io.imread(taj_path)
        taj_color = img_as_float(taj_color)
        
        original_sharpness = analyze_color_image_sharpness(taj_color)
        print(f"Image shape: {taj_color.shape}")
        print(f"Image range: [{taj_color.min():.3f}, {taj_color.max():.3f}]")
        print(f"Original image sharpness score: {original_sharpness:.4f}")
        
        test_params = [
            (1.0, 0.3),   
            (1.0, 0.5),  
            (1.5, 0.4),   
            (2.0, 0.3),   
            (1.0, 0.8),   
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(taj_color)
        axes[0, 0].set_title(f'Original Color\nSharpness: {original_sharpness:.3f}')
        axes[0, 0].axis('off')
        
        for i, (sigma, alpha) in enumerate(test_params):
            if i >= 5:  
                break
                
            sharpened, _, _ = unsharp_masking_color(taj_color, sigma, alpha)
            new_sharpness = analyze_color_image_sharpness(sharpened)
            improvement = new_sharpness - original_sharpness
            
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            axes[row, col].imshow(sharpened)
            axes[row, col].set_title(f'σ={sigma}, α={alpha}\nSharp: {new_sharpness:.3f}\nΔ: {improvement:+.3f}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

        show_color_breakdown(taj_color, sigma=1.5, alpha=0.4)
        compare_color_alpha_values(taj_color)
        
    except FileNotFoundError:
        print(f"Could not find {taj_path}")

def show_color_breakdown(image, sigma=1.5, alpha=0.4):
    sharpened, blurred, high_freq = unsharp_masking_color(image, sigma, alpha)
    sharpened_single, unsharp_kernel = unsharp_masking_color_single_conv(image, sigma, alpha)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Color Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred)
    axes[0, 1].set_title(f'Blurred (σ={sigma})')
    axes[0, 1].axis('off')

    high_freq_vis = high_freq - np.min(high_freq)
    high_freq_vis = high_freq_vis / np.max(high_freq_vis)
    axes[0, 2].imshow(high_freq_vis)
    axes[0, 2].set_title('High Frequencies\n(Scaled for Display)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(sharpened)
    axes[1, 0].set_title(f'Sharpened (α={alpha})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sharpened_single)
    axes[1, 1].set_title('Single Convolution Result')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(unsharp_kernel, cmap='RdBu_r')
    axes[1, 2].set_title('Unsharp Mask Kernel')
    axes[1, 2].axis('off')

    for i, (color, title) in enumerate(zip(['red', 'green', 'blue'], ['Red Channel HF', 'Green Channel HF', 'Blue Channel HF'])):
        hf_channel = high_freq[:, :, i]
        hf_std = np.std(hf_channel)
        axes[2, i].imshow(hf_channel, cmap='RdBu_r', vmin=-3*hf_std, vmax=3*hf_std)
        axes[2, i].set_title(title)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

    difference = np.abs(sharpened - sharpened_single)
    max_diff = np.max(difference)
    print(f"Maximum difference between two-step and single convolution: {max_diff:.6f}")

def compare_color_alpha_values(image):
    sigma = 1.0
    alphas = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, alpha in enumerate(alphas):
        row = i // 3
        col = i % 3
        
        if alpha == 0.0:
            result = image  
            title = f'α={alpha} (Original)'
        else:
            result, _, _ = unsharp_masking_color(image, sigma, alpha)
            title = f'α={alpha}'
        
        axes[row, col].imshow(result)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def color_blur_and_resharpen_test():
    taj_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/taj.jpg'
    
    try:
        image = io.imread(taj_path)
        image = img_as_float(image)

        blur_sigma = 2.0
        blur_kernel_size = int(6 * blur_sigma) + 1
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        
        blur_kernel = create_gaussian_kernel(blur_kernel_size, blur_sigma)

        blurred_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            blurred_image[:, :, channel] = convolve2d(image[:, :, channel], blur_kernel, 
                                                    mode='same', boundary='symm')

        resharpened, _, _ = unsharp_masking_color(blurred_image, sigma=1.0, alpha=0.8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Sharp Color')
        axes[0].axis('off')
        
        axes[1].imshow(blurred_image)
        axes[1].set_title(f'Blurred (σ={blur_sigma})')
        axes[1].axis('off')
        
        axes[2].imshow(resharpened)
        axes[2].set_title('Re-sharpened (α=0.8)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print("Could not find taj.jpg")

if __name__ == "__main__":
    process_color_taj()
    #color_blur_and_resharpen_test()