import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def create_gaussian_stack(image, num_levels=6, sigma_base=1.0):
    gaussian_stack = []
    
    # Level 0
    gaussian_stack.append(image.copy())
    
    # Subsequent levels
    for level in range(1, num_levels):
        sigma = sigma_base * (2 ** level)
        
        if len(image.shape) == 3: 
            blurred = np.zeros_like(image)
            for channel in range(3):
                blurred[:, :, channel] = gaussian_filter(image[:, :, channel], sigma=sigma)
        else: 
            blurred = gaussian_filter(image, sigma=sigma)
            
        gaussian_stack.append(blurred)
    
    return gaussian_stack

def create_laplacian_stack(gaussian_stack):
    laplacian_stack = []
    num_levels = len(gaussian_stack)
    
    # Band-pass levels: L_i = G_i - G_{i+1}
    for i in range(num_levels - 1):
        laplacian = gaussian_stack[i] - gaussian_stack[i + 1]
        laplacian_stack.append(laplacian)

    laplacian_stack.append(gaussian_stack[-1])
    
    return laplacian_stack

def reconstruct_from_laplacian(laplacian_stack):
    reconstructed = np.zeros_like(laplacian_stack[0])
    
    for level in laplacian_stack:
        reconstructed += level
    
    return reconstructed

def visualize_stacks(gaussian_stack, laplacian_stack, title="Gaussian and Laplacian Stacks"):
    num_levels = len(gaussian_stack)
    fig, axes = plt.subplots(2, num_levels, figsize=(3*num_levels, 6))
    
    if num_levels == 1:
        axes = axes.reshape(2, 1)
    
    # Gaussian
    for i in range(num_levels):
        img = gaussian_stack[i]
        if len(img.shape) == 3:
            axes[0, i].imshow(np.clip(img, 0, 1))
        else:
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Gaussian {i}')
        axes[0, i].axis('off')
    
    # Laplacian
    for i in range(num_levels):
        img = laplacian_stack[i]
        
        if i < num_levels - 1:  
            img_display = img + 0.5
            img_display = np.clip(img_display, 0, 1)
        else:  
            img_display = np.clip(img, 0, 1)
        
        if len(img.shape) == 3:
            axes[1, i].imshow(img_display)
        else:
            axes[1, i].imshow(img_display, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Laplacian {i}')
        axes[1, i].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def test_reconstruction(image, num_levels=5):
    print(f"Testing reconstruction with {num_levels} levels...")
 
    gaussian_stack = create_gaussian_stack(image, num_levels)
    laplacian_stack = create_laplacian_stack(gaussian_stack)
    reconstructed = reconstruct_from_laplacian(laplacian_stack)

    error = np.mean(np.abs(image - reconstructed))
    max_error = np.max(np.abs(image - reconstructed))
    
    print(f"Mean reconstruction error: {error:.8f}")
    print(f"Max reconstruction error: {max_error:.8f}")
    print("Perfect reconstruction achieved!" if error < 1e-10 else "Small numerical errors present")

    visualize_stacks(gaussian_stack, laplacian_stack, 
                    f"Test Image - {num_levels} Levels (Error: {error:.2e})")
    
    return gaussian_stack, laplacian_stack, reconstructed

def create_oraple(apple_img, orange_img, num_levels=6):
    h, w = min(apple_img.shape[0], orange_img.shape[0]), min(apple_img.shape[1], orange_img.shape[1])
    apple = apple_img[:h, :w]
    orange = orange_img[:h, :w]
    
    # left half = apple (1), right half = orange (0)
    mask = np.zeros((h, w))
    mask[:, :w//2] = 1

    apple_gaussian = create_gaussian_stack(apple, num_levels)
    orange_gaussian = create_gaussian_stack(orange, num_levels)
    mask_gaussian = create_gaussian_stack(mask, num_levels)
    apple_laplacian = create_laplacian_stack(apple_gaussian)
    orange_laplacian = create_laplacian_stack(orange_gaussian)
    
    # Blend Laplacian levels using Gaussian mask
    blended_laplacian = []
    for level in range(num_levels):
        mask_level = mask_gaussian[level]
        
        if len(apple.shape) == 3: 
            blended_level = np.zeros_like(apple_laplacian[level])
            for c in range(3):
                blended_level[:, :, c] = (
                    mask_level * apple_laplacian[level][:, :, c] +
                    (1 - mask_level) * orange_laplacian[level][:, :, c]
                )
        else:  
            blended_level = (
                mask_level * apple_laplacian[level] +
                (1 - mask_level) * orange_laplacian[level]
            )
        
        blended_laplacian.append(blended_level)

    blended = reconstruct_from_laplacian(blended_laplacian)

    process_imgs = {
        'apple_gaussian': apple_gaussian,
        'orange_gaussian': orange_gaussian,
        'mask_gaussian': mask_gaussian,
        'apple_laplacian': apple_laplacian,
        'orange_laplacian': orange_laplacian,
        'blended_laplacian': blended_laplacian,
        'mask': mask
    }
    
    return blended, process_imgs

def demonstrate_oraple_process(apple_path, orange_path):

    try:
        apple = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/spline/apple.jpeg')
        orange = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/spline/orange.jpeg')
        
        if apple.max() > 1:
            apple = apple / 255.0
        if orange.max() > 1:
            orange = orange / 255.0
        
        print("Creating Oraple blend...")
        blended, process = create_oraple(apple, orange, num_levels=6)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(apple[:blended.shape[0], :blended.shape[1]])
        axes[0, 0].set_title('Apple')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(orange[:blended.shape[0], :blended.shape[1]])
        axes[0, 1].set_title('Orange')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(np.clip(blended, 0, 1))
        axes[0, 2].set_title('Oraple (Blended)')
        axes[0, 2].axis('off')

        for i in range(3):
            if i < len(process['blended_laplacian']) - 1:
                level_img = process['blended_laplacian'][i] + 0.5
            else:
                level_img = process['blended_laplacian'][i]
            
            axes[1, i].imshow(np.clip(level_img, 0, 1))
            axes[1, i].set_title(f'Blended Laplacian Level {i}')
            axes[1, i].axis('off')
        
        plt.suptitle('Multiresolution Blending: The Oraple', fontsize=16)
        plt.tight_layout()
        plt.show()

        print("\nVisualizing Apple Laplacian Stack:")
        visualize_stacks(process['apple_gaussian'], process['apple_laplacian'], "Apple - Gaussian and Laplacian Stacks")
        
        print("Visualizing Orange Laplacian Stack:")
        visualize_stacks(process['orange_gaussian'], process['orange_laplacian'], "Orange - Gaussian and Laplacian Stacks")
        
        return blended, process
        
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        print("Please ensure apple.jpg and orange.jpg are in the current directory")
        return None, None

if __name__ == "__main__":
    blended, process = demonstrate_oraple_process('apple.jpg', 'orange.jpg')