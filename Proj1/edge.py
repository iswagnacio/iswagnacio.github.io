import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.transform import rescale

def downsample_image(image, factor=2):
    return rescale(image, 1.0/factor, anti_aliasing=True, preserve_range=True)

def sobel(image):
    '''sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])'''

    padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')

    grad_x = (
        -1 * padded[:-2, :-2] + 1 * padded[:-2, 2:] +  
        -2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:] +  
        -1 * padded[2:, :-2] + 1 * padded[2:, 2:]        
    )

    grad_y = (
        -1 * padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - 1 * padded[:-2, 2:] +   
        1 * padded[2:, :-2] + 2 * padded[2:, 1:-1] + 1 * padded[2:, 2:]        
    )

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return magnitude

def align_with_edge(channel_to_align, reference_channel, search_range=15):
    ref_edges = sobel(reference_channel)
    channel_edges = sobel(channel_to_align)
    best_score = -1
    best_displacement = (0, 0)

    margin_x = reference_channel.shape[0] // 10
    margin_y = reference_channel.shape[1] // 10
    
    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            shifted_edges = np.roll(channel_edges, dx, axis=1)
            shifted_edges = np.roll(shifted_edges, dy, axis=0)

            ref_inner = ref_edges[margin_x:-margin_x, margin_y:-margin_y]
            shifted_inner = shifted_edges[margin_x:-margin_x, margin_y:-margin_y]

            ref_flat = ref_inner.flatten()
            shifted_flat = shifted_inner.flatten()

            ref_centered = ref_flat - np.mean(ref_flat)
            shifted_centered = shifted_flat - np.mean(shifted_flat)

            numerator = np.dot(ref_centered, shifted_centered)
            denominator = np.linalg.norm(ref_centered) * np.linalg.norm(shifted_centered)
            
            if denominator > 0:
                ncc_score = numerator / denominator
                if ncc_score > best_score:
                    best_score = ncc_score
                    best_displacement = (dx, dy)

    aligned_channel = np.roll(channel_to_align, best_displacement[0], axis=1)
    aligned_channel = np.roll(aligned_channel, best_displacement[1], axis=0)
    
    return best_displacement[0], best_displacement[1], aligned_channel

def align_pyramid(channel_to_align, reference_channel, max_levels=7, base_search_range=25):

    ref_pyramid = [reference_channel]
    channel_pyramid = [channel_to_align]
    current_ref = reference_channel
    current_channel = channel_to_align
    
    for level in range(max_levels):
        if min(current_ref.shape) < 25:
            break
            
        current_ref = downsample_image(current_ref, 2)
        current_channel = downsample_image(current_channel, 2)
        ref_pyramid.append(current_ref)
        channel_pyramid.append(current_channel)

    total_dx, total_dy = 0, 0

    for level in range(len(ref_pyramid) - 1, -1, -1):
        current_search_range = base_search_range if level == len(ref_pyramid) - 1 else base_search_range // 2

        if level < len(ref_pyramid) - 1:
            total_dx *= 2
            total_dy *= 2

            current_channel_shifted = np.roll(channel_pyramid[level], total_dx, axis=1)
            current_channel_shifted = np.roll(current_channel_shifted, total_dy, axis=0)
        else:
            current_channel_shifted = channel_pyramid[level]

        dx, dy, _ = align_with_edge(
            current_channel_shifted, 
            ref_pyramid[level], 
            search_range=current_search_range
        )

        total_dx += dx
        total_dy += dy

    final_aligned = np.roll(channel_to_align, total_dx, axis=1)
    final_aligned = np.roll(final_aligned, total_dy, axis=0)
    
    return total_dx, total_dy, final_aligned

def colorize_image(image_path):

    im = skio.imread(image_path)
    im = sk.img_as_float(im)

    height = im.shape[0] // 3
    b = im[:height]                  
    g = im[height:2*height]           
    r = im[2*height:3*height]

    g_dx, g_dy, aligned_g = align_pyramid(g, b)
    r_dx, r_dy, aligned_r = align_pyramid(r, b)

    color_image = np.dstack([aligned_r, aligned_g, b])
    plt.figure(figsize=(12, 8))
    plt.imshow(color_image)
    plt.title(f'Colorized Image\nG: ({g_dx}, {g_dy}), R: ({r_dx}, {r_dy})')
    plt.axis('off')
    plt.show()

    return color_image, (g_dx, g_dy), (r_dx, r_dy)

if __name__ == "__main__":
    #small_image = '/Users/junwei/Fall2025/CS180/cs180 proj1 data/tobolsk.jpg'
    #result_small = colorize_image(small_image)

    large_image = '/Users/junwei/Fall2025/CS180/cs180 proj1 data/church.tif'
    result_large = colorize_image(large_image)