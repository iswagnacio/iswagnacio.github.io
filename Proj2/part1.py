import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import skimage as sk
import skimage.io as skio
import time

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

def test_convolutions():
    selfie_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/cameraman.png'
    try:
        selfie = plt.imread(selfie_path)

        if len(selfie.shape) == 3:
            selfie = 0.299 * selfie[:,:,0] + 0.587 * selfie[:,:,1] + 0.114 * selfie[:,:,2]

        if selfie.max() > 1.0:
            selfie = selfie.astype(np.float64) / 255.0
            print("✓ Normalized")
        
    except Exception as e:
        print(f"Error loading {selfie_path}: {e}")
        print("Using random test image instead")
        selfie = np.random.rand(100, 100)

    box_filter = np.ones((9, 9)) / 81 
    Dx = np.array([[1, 0, -1]])       
    Dy = np.array([[1], [0], [-1]])   
    
    start = time.time()
    result_4loop_box = convolution_four_loops(selfie, box_filter)
    time_4loop = time.time() - start
    
    start = time.time()
    result_2loop_box = convolution_two_loops(selfie, box_filter)
    time_2loop = time.time() - start
    
    start = time.time()
    result_scipy_box = convolve2d(selfie, box_filter, mode='same', boundary='fill')
    time_scipy = time.time() - start
    
    print(f"  Four loops: {time_4loop:.4f}s")
    print(f"  Two loops:  {time_2loop:.4f}s") 
    print(f"  SciPy:      {time_scipy:.4f}s")
    print(f"  Speedup:    {time_4loop/time_2loop:.1f}x")

    print("\nDx Filter:")
    result_4loop_dx = convolution_four_loops(selfie, Dx)
    result_2loop_dx = convolution_two_loops(selfie, Dx)
    result_scipy_dx = convolve2d(selfie, Dx, mode='same', boundary='fill')

    print("Dy Filter:")
    result_4loop_dy = convolution_four_loops(selfie, Dy)
    result_2loop_dy = convolution_two_loops(selfie, Dy)
    result_scipy_dy = convolve2d(selfie, Dy, mode='same', boundary='fill')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0,0].imshow(selfie, cmap='gray')
    axes[0,0].set_title('Original Selfie')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(result_2loop_box, cmap='gray') 
    axes[0,1].set_title('9x9 Box Filter\n(Blurred)')
    axes[0,1].axis('off')

    axes[1,0].imshow(result_4loop_box, cmap='gray')
    axes[1,0].set_title('4loop Filter')
    axes[1,0].axis('off')

    axes[1,0].imshow(result_scipy_box, cmap='gray')
    axes[1,0].set_title('scipy Filter')
    axes[1,0].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_convolutions()