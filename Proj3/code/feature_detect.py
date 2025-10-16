import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from harris import get_harris_corners, dist2

# B.1
def detect_corners_with_anms(image, num_features=500, edge_discard=20):

    if len(image.shape) == 3:
        raise ValueError("Image must be grayscale! Convert to grayscale before calling.")
    if image.dtype == np.uint8:
        image_float = image.astype(np.float64) / 255.0
    else:
        image_float = image.astype(np.float64)
        if image_float.max() > 1.0:
            image_float = image_float / 255.0
    
    h, coords_all = get_harris_corners(image_float, edge_discard=edge_discard)
    coords_anms = adaptive_non_maximal_suppression(coords_all, h, num_features=num_features)
    
    return coords_all, coords_anms, h

def adaptive_non_maximal_suppression(coords, h, num_features=500, c_robust=0.9):
    
    corner_strengths = h[coords[0], coords[1]]
    n = coords.shape[1]
    radii = np.inf * np.ones(n)
    
    for i in range(n):
        stronger_mask = corner_strengths > c_robust * corner_strengths[i]
        
        if np.any(stronger_mask):
            stronger_coords = coords[:, stronger_mask]
            point_i = coords[:, i].reshape(1, 2)  
            stronger_points = stronger_coords.T  
            distances_sq = dist2(point_i, stronger_points)
            radii[i] = np.sqrt(np.min(distances_sq))
    
    sorted_indices = np.argsort(radii)[::-1]
    num_to_select = min(num_features, n)
    selected_indices = sorted_indices[:num_to_select]
    selected_coords = coords[:, selected_indices]
    
    return selected_coords

def visualize_harris_corners(image, coords_all, coords_anms=None, save_path=None):

    if coords_anms is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[0].scatter(coords_all[1], coords_all[0], c='red', s=3, marker='x', alpha=0.5)
        axes[0].set_title(f'All Harris Corners ({coords_all.shape[1]} points)')
        axes[0].axis('off')

        axes[1].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[1].scatter(coords_anms[1], coords_anms[0], c='blue', s=30, marker='o', 
                       facecolors='none', edgecolors='blue', linewidths=2)
        axes[1].set_title(f'ANMS Selected Corners ({coords_anms.shape[1]} points)')
        axes[1].axis('off')

    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        ax.scatter(coords_all[1], coords_all[0], c='red', s=5, marker='x')
        ax.set_title(f'All Harris Corners ({coords_all.shape[1]} points)')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def test_harris_anms(image_path, num_features=500, output_dir='.'):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coords_all, coords_anms, h = detect_corners_with_anms(
        img_gray, 
        num_features=num_features,
        edge_discard=20
    )

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f'{base_name}_harris_anms.png')
    visualize_harris_corners(img_rgb, coords_all, coords_anms, save_path=save_path)
    
    return coords_all, coords_anms, h, img_gray, img_rgb

#B.2
def extract_feature_descriptors(image, coords, descriptor_size=8, sample_spacing=5):

    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    else:
        image = image.astype(np.float64)
    
    h, w = image.shape
    n_features = coords.shape[1]
    sigma = sample_spacing / 2.0
    blurred = gaussian_filter(image, sigma=sigma)

    descriptors = []
    valid_coords = []
    margin = (descriptor_size * sample_spacing) / 2.0
    
    for i in range(n_features):
        y, x = coords[:, i]
        
        if (x < margin or x >= w - margin or 
            y < margin or y >= h - margin):
            continue
        
        descriptor = np.zeros((descriptor_size, descriptor_size))
        
        for row in range(descriptor_size):
            for col in range(descriptor_size):
                offset = (descriptor_size - 1) / 2.0
                sample_y = y + (row - offset) * sample_spacing
                sample_x = x + (col - offset) * sample_spacing
                
                y0 = int(np.floor(sample_y))
                x0 = int(np.floor(sample_x))
                y1 = y0 + 1
                x1 = x0 + 1
                
                if y0 < 0 or y1 >= h or x0 < 0 or x1 >= w:
                    break
                
                wy = sample_y - y0
                wx = sample_x - x0
                
                descriptor[row, col] = (
                    (1 - wy) * (1 - wx) * blurred[y0, x0] +
                    (1 - wy) * wx * blurred[y0, x1] +
                    wy * (1 - wx) * blurred[y1, x0] +
                    wy * wx * blurred[y1, x1]
                )
        
        desc_vector = descriptor.flatten()
        mean = np.mean(desc_vector)
        std = np.std(desc_vector)
        
        if std < 1e-10:
            continue
        
        desc_normalized = (desc_vector - mean) / std
        
        descriptors.append(desc_normalized)
        valid_coords.append(coords[:, i])
    
    if len(descriptors) == 0:
        return np.array([]), np.array([]).reshape(2, 0)
    
    descriptors = np.array(descriptors)
    valid_coords = np.array(valid_coords).T

    return descriptors, valid_coords

def visualize_descriptors(image, coords, descriptors, num_to_show=10, save_path=None):
 
    num_to_show = min(num_to_show, coords.shape[1])
    indices = np.linspace(0, coords.shape[1] - 1, num_to_show, dtype=int)
    
    fig = plt.figure(figsize=(20, 6))
    
    for idx, i in enumerate(indices):
        ax = plt.subplot(3, num_to_show, idx + 1)
        desc_patch = descriptors[i].reshape(8, 8)
        ax.imshow(desc_patch, cmap='gray', interpolation='nearest')
        ax.set_title(f'Desc {i}', fontsize=10)
        ax.axis('off')
        
        ax = plt.subplot(3, num_to_show, num_to_show + idx + 1)
        y, x = coords[:, i]
        margin = 40
        y0 = max(0, int(y - margin))
        y1 = min(image.shape[0], int(y + margin))
        x0 = max(0, int(x - margin))
        x1 = min(image.shape[1], int(x + margin))
        
        patch = image[y0:y1, x0:x1]
        ax.imshow(patch, cmap='gray' if len(image.shape) == 2 else None)

        local_y = y - y0
        local_x = x - x0
        ax.plot(local_x, local_y, 'r+', markersize=12, markeredgewidth=2)
        rect_size = 40
        rect = plt.Rectangle((local_x - rect_size/2, local_y - rect_size/2), 
                            rect_size, rect_size,
                            fill=False, edgecolor='yellow', linewidth=1.5)
        ax.add_patch(rect)
        
        ax.set_title(f'({int(x)}, {int(y)})', fontsize=10)
        ax.axis('off')
        
        ax = plt.subplot(3, num_to_show, 2*num_to_show + idx + 1)
        ax.bar(range(64), descriptors[i], width=1.0)
        ax.set_ylim([-3, 3])
        ax.set_title('Normalized', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([-3, 0, 3])
        ax.axhline(0, color='red', linewidth=0.5)
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def simple_corner_selection(image, num_features=500):

    if image.dtype == np.uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

    h, coords = get_harris_corners(img_float, edge_discard=20)
    corner_strengths = h[coords[0], coords[1]]

    if coords.shape[1] > num_features:
        sorted_idx = np.argsort(corner_strengths)[::-1]
        selected = sorted_idx[:num_features]
        coords = coords[:, selected]

    return coords, h

def test_descriptors(image_path, num_features=500, output_dir='.'):

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords, h = simple_corner_selection(img_gray, num_features=num_features)

    descriptors, valid_coords = extract_feature_descriptors(
        img_gray, coords,
        descriptor_size=8,
        sample_spacing=5
    )

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f'{base_name}_descriptors.png')
    
    if descriptors.shape[0] > 0:
        visualize_descriptors(img_rgb, valid_coords, descriptors, 
                            num_to_show=10, save_path=save_path)
    
    return valid_coords, descriptors, img_gray, img_rgb

#B.3
def match_features(descriptors1, descriptors2, ratio_threshold=0.8):

    dist_matrix = cdist(descriptors1, descriptors2, metric='euclidean')
    matches = []
    match_distances = []
    
    for i in range(descriptors1.shape[0]):
        distances = dist_matrix[i, :]
        sorted_indices = np.argsort(distances)
        
        if len(sorted_indices) < 2:
            continue
        
        nearest_idx = sorted_indices[0]
        second_nearest_idx = sorted_indices[1]
        
        dist_1nn = distances[nearest_idx]
        dist_2nn = distances[second_nearest_idx]
        
        if dist_2nn > 1e-10: 
            ratio = dist_1nn / dist_2nn
            
            if ratio < ratio_threshold:
                matches.append([i, nearest_idx])
                match_distances.append(dist_1nn)
    
    matches = np.array(matches) if matches else np.array([]).reshape(0, 2)
    match_distances = np.array(match_distances) if match_distances else np.array([])
    
    return matches, match_distances

def visualize_matches(img1, img2, coords1, coords2, matches, 
                     max_to_show=50, save_path=None):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    h_max = max(h1, h2)
    if len(img1.shape) == 2:
        combined = np.zeros((h_max, w1 + w2), dtype=img1.dtype)
    else:
        combined = np.zeros((h_max, w1 + w2, 3), dtype=img1.dtype)
    
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2

    num_to_show = min(max_to_show, len(matches))
    if num_to_show > 0:
        indices = np.random.choice(len(matches), num_to_show, replace=False)
        indices = sorted(indices)
        selected_matches = matches[indices]
    else:
        selected_matches = []

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(combined, cmap='gray' if len(combined.shape) == 2 else None)
    colors = plt.cm.hsv(np.linspace(0, 1, num_to_show))
    
    for idx, (i, j) in enumerate(selected_matches):
        y1, x1 = coords1[:, i]
        y2, x2 = coords2[:, j]
        x2_offset = x2 + w1
        
        ax.plot([x1, x2_offset], [y1, y2], 
               color=colors[idx], linewidth=1.5, alpha=0.6)
        ax.plot(x1, y1, 'o', color=colors[idx], markersize=6, 
               markerfacecolor='none', markeredgewidth=2)
        ax.plot(x2_offset, y2, 'o', color=colors[idx], markersize=6,
               markerfacecolor='none', markeredgewidth=2)
    
    ax.set_title(f'Feature Matches (showing {num_to_show} of {len(matches)} total matches)', 
                fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    plt.show()

def match_image_pair(image_path1, image_path2, num_features=500, 
                    ratio_threshold=0.8, output_dir='.'):

    coords1, descriptors1, img_gray1, img_rgb1 = test_descriptors(
        image_path1, num_features=num_features, output_dir=output_dir
    )
    coords2, descriptors2, img_gray2, img_rgb2 = test_descriptors(
        image_path2, num_features=num_features, output_dir=output_dir
    )

    matches, distances = match_features(
        descriptors1, descriptors2, 
        ratio_threshold=ratio_threshold
    )
    
    if len(matches) > 0:
        print(f"Match distance statistics:")
        print(f"  Mean:   {np.mean(distances):.4f}")
        print(f"  Median: {np.median(distances):.4f}")
        print(f"  Min:    {np.min(distances):.4f}")
        print(f"  Max:    {np.max(distances):.4f}")

    if len(matches) > 0:
        base_name1 = os.path.splitext(os.path.basename(image_path1))[0]
        base_name2 = os.path.splitext(os.path.basename(image_path2))[0]
        match_save_path = os.path.join(output_dir, 
                                       f'{base_name1}_{base_name2}_matches.png')
        
        visualize_matches(img_rgb1, img_rgb2, coords1, coords2, matches,
                         max_to_show=50, save_path=match_save_path)
    else:
        print("\n⚠️  No matches found!")
    
    return matches, coords1, coords2, descriptors1, descriptors2
