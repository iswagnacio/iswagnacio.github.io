from matplotlib import pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as skio

def align_single_scale(channel_to_align, reference_channel, search_range=15):
    best_score = -1
    best_displacement = (0, 0)

    margin_x = reference_channel.shape[0] // 10
    margin_y = reference_channel.shape[1] // 10
    
    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            shifted_channel = np.roll(channel_to_align, dx, axis=1)
            shifted_channel = np.roll(shifted_channel, dy, axis=0)

            ref_inner = reference_channel[margin_x:-margin_x, margin_y:-margin_y]
            shifted_inner = shifted_channel[margin_x:-margin_x, margin_y:-margin_y]

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

def colorize_image(image_path):
    im = skio.imread(image_path)
    im = sk.img_as_float(im)

    height = im.shape[0] // 3
    b = im[:height]       
    g = im[height:2*height]            
    r = im[2*height:3*height]

    g_dx, g_dy, aligned_g = align_single_scale(g, b)
    r_dx, r_dy, aligned_r = align_single_scale(r, b)

    color_image = np.dstack([aligned_r, aligned_g, b])
    plt.figure(figsize=(12, 8))
    plt.imshow(color_image)
    plt.axis('off')
    plt.show()

    return color_image

result = colorize_image('/Users/junwei/Fall2025/CS180/cs180 proj1 data/tobolsk.jpg')