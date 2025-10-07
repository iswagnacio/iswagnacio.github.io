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
    threshold = np.percentile(gradient_magnitude, 91)
    edge_image = gradient_magnitude > threshold

    output_dir = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2'
    import os
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(dx, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dx_partial_derivative.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(dy, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dy_partial_derivative.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gradient_magnitude.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(edge_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/binarized_edges.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    percentiles = [85, 88, 91, 94, 97, 99]
    
    for i, p in enumerate(percentiles):
        thresh = np.percentile(gradient_magnitude, p)
        edges = gradient_magnitude > thresh
        
        row = i // 3
        col = i % 3
        axes[row, col].imshow(edges, cmap='gray')
        axes[row, col].set_title(f'{p}th percentile\n(threshold={thresh:.3f})', fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/threshold_comparison.png', dpi=150, bbox_inches='tight')
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

    dog_x = convolve2d(gaussian_2d, Dx, mode='same', boundary='fill')
    dog_y = convolve2d(gaussian_2d, Dy, mode='same', boundary='fill')
    dx_method2 = convolve2d(image, dog_x, mode='same', boundary='fill')
    dy_method2 = convolve2d(image, dog_y, mode='same', boundary='fill')
    gradient_mag_method2 = np.sqrt(dx_method2**2 + dy_method2**2)
    
    threshold2 = np.percentile(gradient_mag_method2, 85)
    edge_method2 = gradient_mag_method2 > threshold2

    output_dir = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2'
    import os
    os.makedirs(output_dir, exist_ok=True)

    # blurred
    plt.figure(figsize=(8, 8))
    plt.imshow(blurred_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    #plt.savefig(f'{output_dir}/cameraman_blurred.png', dpi=150, bbox_inches='tight')
    #plt.close()
    
    # Gaussian filter visualization sigma1.7 size15*15
    plt.figure(figsize=(8, 8))
    plt.imshow(gaussian_2d, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gaussian_filter_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # DoG filters
    plt.figure(figsize=(8, 8))
    plt.imshow(dog_x, cmap='RdBu_r')
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dog_x_filter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(dog_y, cmap='RdBu_r')
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dog_y_filter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Method 1 results (blur first)
    plt.figure(figsize=(8, 8))
    plt.imshow(gradient_mag_method1, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sequent_gradient_mag.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    #threshold 0.062
    plt.figure(figsize=(8, 8))
    plt.imshow(edge_method1, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sequent_edges.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Method 2 results (DoG)
    plt.figure(figsize=(8, 8))
    plt.imshow(gradient_mag_method2, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/DoG_gradient_mag.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    #threshold 0.061
    plt.figure(figsize=(8, 8))
    plt.imshow(edge_method2, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/DoG_edges.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'method1': {'dx': dx_method1, 'dy': dy_method1, 'mag': gradient_mag_method1, 'edges': edge_method1},
        'method2': {'dx': dx_method2, 'dy': dy_method2, 'mag': gradient_mag_method2, 'edges': edge_method2},
        'filters': {'gaussian': gaussian_2d, 'dog_x': dog_x, 'dog_y': dog_y},
        'blurred': blurred_image
    }

def test_convolutions():
    selfie_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/DSC_2237.jpg'
    selfie = read_image(selfie_path)
    box_filter = create_box_filter(9)  

    #4 loops
    start_time = time.time()
    result_4loop_box = convolution_four_loops(selfie, box_filter)
    time_4loop = time.time() - start_time
    
    # 2 loops
    start_time = time.time()
    result_2loop_box = convolution_two_loops(selfie, box_filter)
    time_2loop = time.time() - start_time
    
    # Scipy
    start_time = time.time()
    result_scipy_box = convolve2d(selfie, box_filter, mode='same', boundary='fill')
    time_scipy = time.time() - start_time
    
    print(f"Four loops runtime: {time_4loop:.4f} seconds")
    print(f"Two loops runtime: {time_2loop:.4f} seconds") 
    print(f"Scipy runtime: {time_scipy:.4f} seconds")
    
    # Finite difference
    result_4loop_dx = convolution_four_loops(selfie, Dx)
    result_4loop_dy = convolution_four_loops(selfie, Dy)
    result_2loop_dx = convolution_two_loops(selfie, Dx) 
    result_2loop_dy = convolution_two_loops(selfie, Dy)
    result_scipy_dx = convolve2d(selfie, Dx, mode='same', boundary='fill')
    result_scipy_dy = convolve2d(selfie, Dy, mode='same', boundary='fill')
    
    output_dir = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2'
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Saving individual result images...")
    
    # Original selfie
    plt.figure(figsize=(8, 8))
    plt.imshow(selfie, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/original_selfie.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Box filter results
    plt.figure(figsize=(8, 8))
    plt.imshow(result_4loop_box, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4loop_box_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(result_2loop_box, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2loop_box_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(result_scipy_box, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scipy_box_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Box filter visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(box_filter, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/box_filter_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dx results
    plt.figure(figsize=(8, 8))
    plt.imshow(result_4loop_dx, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4loop_dx_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(result_2loop_dx, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2loop_dx_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(result_scipy_dx, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scipy_dx_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dy results
    plt.figure(figsize=(8, 8))
    plt.imshow(result_4loop_dy, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4loop_dy_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(result_2loop_dy, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2loop_dy_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(result_scipy_dy, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scipy_dy_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dx filter visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(Dx, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dx_filter_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dy filter visualization  
    plt.figure(figsize=(6, 6))
    plt.imshow(Dy, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dy_filter_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dx comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(selfie, cmap='gray')
    axes[0, 0].set_title('Original Selfie', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(result_4loop_dx, cmap='gray')
    axes[0, 1].set_title('4-Loop Dx', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(result_2loop_dx, cmap='gray')
    axes[1, 0].set_title('2-Loop Dx', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(result_scipy_dx, cmap='gray')
    axes[1, 1].set_title('Scipy Dx', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dx_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dy comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(selfie, cmap='gray')
    axes[0, 0].set_title('Original Selfie', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(result_4loop_dy, cmap='gray')
    axes[0, 1].set_title('4-Loop Dy', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(result_2loop_dy, cmap='gray')
    axes[1, 0].set_title('2-Loop Dy', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(result_scipy_dy, cmap='gray')
    axes[1, 1].set_title('Scipy Dy', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    h, w = selfie.shape
    corner_size = 50
    
    # Top-left corner
    original_corner = selfie[:corner_size, :corner_size]
    scipy_corner = result_scipy_box[:corner_size, :corner_size]
    custom_corner = result_4loop_box[:corner_size, :corner_size]
    
    axes[0, 0].imshow(original_corner, cmap='gray')
    axes[0, 0].set_title('Original (Top-left corner)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(custom_corner, cmap='gray')
    axes[0, 1].set_title('Custom Conv (zero padding)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(scipy_corner, cmap='gray')
    axes[0, 2].set_title('Scipy Conv (fill boundary)', fontsize=12)
    axes[0, 2].axis('off')
    
    # Bottom-right corner
    original_corner_br = selfie[-corner_size:, -corner_size:]
    scipy_corner_br = result_scipy_box[-corner_size:, -corner_size:]
    custom_corner_br = result_4loop_box[-corner_size:, -corner_size:]
    
    axes[1, 0].imshow(original_corner_br, cmap='gray')
    axes[1, 0].set_title('Original (Bottom-right corner)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(custom_corner_br, cmap='gray')
    axes[1, 1].set_title('Custom Conv (zero padding)', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(scipy_corner_br, cmap='gray')
    axes[1, 2].set_title('Scipy Conv (fill boundary)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boundary_handling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Performance visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    methods = ['4-Loop', '2-Loop', 'Scipy']
    times = [time_4loop, time_2loop, time_scipy]
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    bars = ax.bar(methods, times, color=colors, alpha=0.8)
    ax.set_ylabel('Runtime (seconds)', fontsize=14)
    ax.set_title('Convolution Performance Comparison', fontsize=16, pad=20)
    ax.set_yscale('log') 

    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.3f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    saved_files = [
        "original_selfie.png",
        "4loop_box_result.png", "2loop_box_result.png", "scipy_box_result.png",
        "4loop_dx_result.png", "2loop_dx_result.png", "scipy_dx_result.png", 
        "4loop_dy_result.png", "2loop_dy_result.png", "scipy_dy_result.png",
        "box_filter_visualization.png", "dx_filter_visualization.png", "dy_filter_visualization.png",
        "box_filter_comparison.png", "dx_comparison.png", "dy_comparison.png",
        "boundary_handling_comparison.png", "performance_comparison.png"
    ]
    
    for filename in saved_files:
        print(f"  - {filename}")
    
    return {
        'results': {
            'box_4loop': result_4loop_box,
            'box_2loop': result_2loop_box, 
            'box_scipy': result_scipy_box,
            'dx_scipy': result_scipy_dx,
            'dy_scipy': result_scipy_dy
        },
        'timing': {
            'four_loops': time_4loop,
            'two_loops': time_2loop,
            'scipy': time_scipy
        }
    }
  

if __name__ == "__main__":
    #test_convolutions()
    #finite_difference('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/cameraman.png')
    gaussian('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/cameraman.png', sigma=1.7, kernel_size=15)