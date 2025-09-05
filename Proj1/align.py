import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.segmentation import flood
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel

def align_single_scale(channel_to_align, reference_channel, search_range=5):
    """
    Align channel_to_align to reference_channel using exhaustive search with NCC
    Returns: (best_dx, best_dy, aligned_channel)
    """
    best_score = -1  # For NCC, higher is better (range -1 to 1)
    best_displacement = (0, 0)

    margin = 60  # Adjust this value

    ref_inner = reference_channel[margin:-margin, margin:-margin]
    channel_inner = channel_to_align[margin:-margin, margin:-margin]

    '''plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(ref_inner, cmap='gray')
    plt.title('Blue')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(channel_inner, cmap='gray')
    plt.title('Green/Red')
    plt.axis('off')

    plt.tight_layout()
    plt.show()'''

    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            shifted = np.roll(channel_to_align, dx, axis=1)
            shifted = np.roll(shifted, dy, axis=0)

            # Only compute metric on inner region (exclude borders)
            ref_inner = reference_channel[margin:-margin, margin:-margin]
            shifted_inner = shifted[margin:-margin, margin:-margin]

            # Compute Normalized Cross-Correlation (NCC)
            # Flatten arrays for easier computation
            ref_flat = ref_inner.flatten()
            shifted_flat = shifted_inner.flatten()
            
            # Subtract means (center the data)
            ref_centered = ref_flat - np.mean(ref_flat)
            shifted_centered = shifted_flat - np.mean(shifted_flat)
            
            # Compute normalized cross-correlation
            numerator = np.dot(ref_centered, shifted_centered)
            denominator = np.linalg.norm(ref_centered) * np.linalg.norm(shifted_centered)
            
            if denominator > 0:
                ncc_score = numerator / denominator
            else:
                ncc_score = -1  # Worst possible score
            
            # For NCC, higher values are better
            if ncc_score > best_score:
                best_score = ncc_score
                best_displacement = (dx, dy)

    print(f"Best NCC score: {best_score:.4f}")
    
    # Apply best displacement
    best_aligned = np.roll(channel_to_align, best_displacement[0], axis=1)
    best_aligned = np.roll(best_aligned, best_displacement[1], axis=0)

    return best_displacement[0], best_displacement[1], best_aligned

def colorize_image(image_path):
    # Read and convert image
    im = skio.imread(image_path)
    im = sk.img_as_float(im)
    im = im[:, 20:-20]
    # Split into BGR channels (top to bottom)
    height = im.shape[0] // 3
    b = im[:height]                    # Blue (reference)
    g = im[height:2*height]            # Green (align to blue)
    r = im[2*height:3*height]          # Red (align to blue)

    # Align G and R to B
    g_dx, g_dy, aligned_g = align_single_scale(g, b)
    r_dx, r_dy, aligned_r = align_single_scale(r, b)

    # Print displacements
    print(f"Green channel displacement: ({g_dx}, {g_dy})")
    print(f"Red channel displacement: ({r_dx}, {r_dy})")

    # Create RGB color image (note: BGR -> RGB conversion)
    color_image = np.dstack([aligned_r, aligned_g, b])
    skio.imshow(color_image)
    skio.show()

    return color_image

color_image = colorize_image('/Users/junwei/Fall2025/CS180/cs180 proj1 data/emir.tif')