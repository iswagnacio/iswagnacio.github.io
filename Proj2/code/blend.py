import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def create_gaussian_stack(image, num_levels=6, sigma_base=1.0):
    gaussian_stack = []

    gaussian_stack.append(image.copy())

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

def visualize_stacks(gaussian_stack, laplacian_stack):
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
    
    plt.tight_layout()
    plt.show()

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

def demonstrate_oraple_process():

    try:
        apple = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/apple.jpeg')
        orange = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/orange.jpeg')
        
        if apple.max() > 1:
            apple = apple / 255.0
        if orange.max() > 1:
            orange = orange / 255.0

        blended, process = create_oraple(apple, orange, num_levels=6)

        plt.imshow(np.clip(blended, 0, 1))
        plt.imsave('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/oraple.png', np.clip(blended, 0, 1))
        plt.axis('off')
        plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i in range(3):
            if i < len(process['blended_laplacian']) - 1:
                level_img = process['blended_laplacian'][i] + 0.5
            else:
                level_img = process['blended_laplacian'][i]
            
            axes[i].imshow(np.clip(level_img, 0, 1))
            axes[i].set_title(f'Blended Laplacian Level {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

        #visualize_stacks(process['apple_gaussian'], process['apple_laplacian'])
        visualize_stacks(process['orange_gaussian'], process['orange_laplacian'])
        
        return blended, process
        
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return None, None

def load_and_process_mask(mask_path):
    mask = plt.imread(mask_path)
    if len(mask.shape) == 3:
        mask = np.mean(mask, axis=2)
    if mask.max() > 1:
        mask = mask / 255.0
    
    return mask

def create_custom_blend(image1, image2, mask_path, num_levels=6):

    mask = load_and_process_mask(mask_path)
    h = min(image1.shape[0], image2.shape[0], mask.shape[0])
    w = min(image1.shape[1], image2.shape[1], mask.shape[1])
    
    img1 = image1[:h, :w]  # Background
    img2 = image2[:h, :w]  # Foreground 
    mask = mask[:h, :w]
    
    # Create Gaussian stacks
    img1_gaussian = create_gaussian_stack(img1, num_levels)
    img2_gaussian = create_gaussian_stack(img2, num_levels)
    mask_gaussian = create_gaussian_stack(mask, num_levels)
    
    # Create Laplacian stacks
    img1_laplacian = create_laplacian_stack(img1_gaussian)
    img2_laplacian = create_laplacian_stack(img2_gaussian)

    blended_laplacian = []
    for level in range(num_levels):
        mask_level = mask_gaussian[level]
        
        if len(img1.shape) == 3: 
            blended_level = np.zeros_like(img1_laplacian[level])
            for c in range(3):
                blended_level[:, :, c] = (
                    (1 - mask_level) * img2_laplacian[level][:, :, c] +
                    mask_level * img1_laplacian[level][:, :, c]
                )
        else: 
            blended_level = (
                (1 - mask_level) * img2_laplacian[level] +
                mask_level * img1_laplacian[level]
            )
        
        blended_laplacian.append(blended_level)

    blended = reconstruct_from_laplacian(blended_laplacian)
    
    return blended

def demonstrate_custom_blend():
    try:
        camera_img = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/DSC_0656.jpg') 
        viewfinder_img = plt.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/DSC_0993.jpeg')
        mask_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj2/media/mask.jpg' 

        if camera_img.max() > 1:
            camera_img = camera_img / 255.0
        if viewfinder_img.max() > 1:
            viewfinder_img = viewfinder_img / 255.0

        blended, process = create_custom_blend(camera_img, viewfinder_img, mask_path, num_levels=6)

        plt.figure(figsize=(10, 8))
        plt.imshow(np.clip(blended, 0, 1))
        plt.axis('off')
        plt.title('Final Custom Blended Image')
        plt.show()
        
        return blended, process
        
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return None, None

if __name__ == "__main__":
    #blended, process = demonstrate_oraple_process()
    blended, process = demonstrate_custom_blend()