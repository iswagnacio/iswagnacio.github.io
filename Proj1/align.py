import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.transform import rescale
#TODO: Apply Canny Edge Detection to splitted images.
#TODO: Apply NCC on edge-detected images.


def downsample_image(image, factor=2):
    """
    Downsample image by a given factor using rescale
    """
    return rescale(image, 1.0/factor, anti_aliasing=True, preserve_range=True)

def align_single_scale(channel_to_align, reference_channel, search_range=15):
    """
    Align channel_to_align to reference_channel using exhaustive search with NCC
    Returns: (best_dx, best_dy, aligned_channel)
    """
    best_score = -1
    best_displacement = (0, 0)

    # Use adaptive margin based on image size
    margin = min(60, min(reference_channel.shape) // 10)

    if margin * 2 >= min(reference_channel.shape):
        margin = max(5, min(reference_channel.shape) // 4)

    ref_inner = reference_channel[margin:-margin, margin:-margin]

    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            shifted = np.roll(channel_to_align, dx, axis=1)
            shifted = np.roll(shifted, dy, axis=0)

            # Only compute metric on inner region (exclude borders)
            shifted_inner = shifted[margin:-margin, margin:-margin]

            # Compute Normalized Cross-Correlation (NCC)
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
                ncc_score = -1

            if ncc_score > best_score:
                best_score = ncc_score
                best_displacement = (dx, dy)

    print(f"Best NCC score: {best_score:.4f}")
    
    # Apply best displacement
    best_aligned = np.roll(channel_to_align, best_displacement[0], axis=1)
    best_aligned = np.roll(best_aligned, best_displacement[1], axis=0)

    return best_displacement[0], best_displacement[1], best_aligned

def align_pyramid(channel_to_align, reference_channel, max_levels=5, base_search_range=15):
    """
    Align two channels using image pyramid for efficiency
    Returns: (total_dx, total_dy, aligned_channel)
    """
    # Build pyramids
    ref_pyramid = [reference_channel]
    channel_pyramid = [channel_to_align]
    
    # Create pyramid levels
    current_ref = reference_channel
    current_channel = channel_to_align
    
    for level in range(max_levels):
        if min(current_ref.shape) < 32:
            break
            
        current_ref = downsample_image(current_ref, 2)
        current_channel = downsample_image(current_channel, 2)
        ref_pyramid.append(current_ref)
        channel_pyramid.append(current_channel)
    
    print(f"Created pyramid with {len(ref_pyramid)} levels")
    for i, img in enumerate(ref_pyramid):
        print(f"Level {i}: {img.shape}")
    
    # Start from coarsest level (last in pyramid)
    total_dx, total_dy = 0, 0
    
    # Work from coarsest to finest
    for level in range(len(ref_pyramid) - 1, -1, -1):
        print(f"\nProcessing pyramid level {level} (shape: {ref_pyramid[level].shape})")
        
        # Scale search range based on pyramid level
        current_search_range = base_search_range if level == len(ref_pyramid) - 1 else base_search_range // 2
        
        # Apply previous displacement estimate (scaled up from coarser level)
        if level < len(ref_pyramid) - 1:
            # Scale up the displacement from the previous (coarser) level
            total_dx *= 2
            total_dy *= 2
            
            # Apply the scaled displacement
            current_channel_shifted = np.roll(channel_pyramid[level], total_dx, axis=1)
            current_channel_shifted = np.roll(current_channel_shifted, total_dy, axis=0)
        else:
            current_channel_shifted = channel_pyramid[level]
        
        # Fine-tune alignment at current level
        dx, dy, _ = align_single_scale(
            current_channel_shifted, 
            ref_pyramid[level], 
            search_range=current_search_range
        )
        
        # Update total displacement
        total_dx += dx
        total_dy += dy
        
        print(f"Level {level} displacement: ({dx}, {dy})")
        print(f"Cumulative displacement: ({total_dx}, {total_dy})")
    
    # Apply final displacement to original full-resolution image
    final_aligned = np.roll(channel_to_align, total_dx, axis=1)
    final_aligned = np.roll(final_aligned, total_dy, axis=0)
    
    return total_dx, total_dy, final_aligned

def colorize_image(image_path, use_pyramid=True, use_edges=False):
    """
    Main function to colorize Prokudin-Gorskii images
    """
    print(f"Processing: {image_path}")
    
    # Read and convert image
    im = skio.imread(image_path)
    im = sk.img_as_float(im)
    
    # Remove some border artifacts
    im = im[:, 20:-20]
    
    # Split into BGR channels (top to bottom)
    height = im.shape[0] // 3
    b = im[:height]                    # Blue (reference)
    g = im[height:2*height]            # Green (align to blue)
    r = im[2*height:3*height]          # Red (align to blue)
    
    print(f"Image dimensions: {im.shape}")
    print(f"Each channel: {b.shape}")

    if use_pyramid and min(b.shape) > 200:
        print("Using pyramid alignment for large image...")
        g_dx, g_dy, aligned_g = align_pyramid(g, b)
        r_dx, r_dy, aligned_r = align_pyramid(r, b)

    # Print final displacements
    print(f"\nFinal Results:")
    print(f"Green channel displacement: ({g_dx}, {g_dy})")
    print(f"Red channel displacement: ({r_dx}, {r_dy})")

    # Create RGB color image (note: BGR -> RGB conversion)
    color_image = np.dstack([aligned_r, aligned_g, b])
    
    # Display result
    plt.figure(figsize=(12, 8))
    plt.imshow(color_image)
    plt.title(f'Colorized Image\nG: ({g_dx}, {g_dy}), R: ({r_dx}, {r_dy})')
    plt.axis('off')
    plt.show()

    return color_image, (g_dx, g_dy), (r_dx, r_dy)

# Example usage following skeleton structure
if __name__ == "__main__":
    # Test with different images
    
    # Small image (use single-scale)
    print("=== PROCESSING SMALL IMAGE ===")
    small_image = '/Users/junwei/Fall2025/CS180/cs180 proj1 data/cathedral.jpg'  # Adjust path
    try:
        result_small = colorize_image(small_image)
    except FileNotFoundError:
        print(f"File not found: {small_image}")
    
    print("\n" + "="*50 + "\n")
    
    # Large image (use pyramid)
    print("=== PROCESSING LARGE IMAGE ===")
    large_image = '/Users/junwei/Fall2025/CS180/cs180 proj1 data/emir.tif'  # Adjust path
    try:
        result_large = colorize_image(large_image)
    except FileNotFoundError:
        print(f"File not found: {large_image}")