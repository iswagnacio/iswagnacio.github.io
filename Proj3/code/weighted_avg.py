import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os
from scipy.ndimage import distance_transform_edt

def computeH(im1_pts, im2_pts):
    N = im1_pts.shape[0]
    mean1 = np.mean(im1_pts, axis=0)
    scale1 = np.sqrt(2) / np.mean(np.linalg.norm(im1_pts - mean1, axis=1))
    T1 = np.array([
        [scale1, 0, -scale1 * mean1[0]],
        [0, scale1, -scale1 * mean1[1]],
        [0, 0, 1]
    ])

    mean2 = np.mean(im2_pts, axis=0)
    scale2 = np.sqrt(2) / np.mean(np.linalg.norm(im2_pts - mean2, axis=1))
    T2 = np.array([
        [scale2, 0, -scale2 * mean2[0]],
        [0, scale2, -scale2 * mean2[1]],
        [0, 0, 1]
    ])

    im1_pts_norm = (T1 @ np.concatenate([im1_pts.T, np.ones((1, N))]))[:2].T
    im2_pts_norm = (T2 @ np.concatenate([im2_pts.T, np.ones((1, N))]))[:2].T
    A = []
    b = []
    
    for i in range(N):
        x, y = im1_pts_norm[i]
        x_prime, y_prime = im2_pts_norm[i]
        A.append([x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime])
        b.append(x_prime)
        A.append([0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime])
        b.append(y_prime)
    
    A = np.array(A)
    b = np.array(b)
    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    H_norm = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ])
    H = np.linalg.inv(T2) @ H_norm @ T1
    
    return H

def load_and_compute_homography(correspondence_data):

    im1_pts = np.array(correspondence_data['im1Points'])
    im2_pts = np.array(correspondence_data['im2Points'])
    
    H = computeH(im1_pts, im2_pts)
    
    return H

def warpImageNearestNeighbor(im, H, output_shape=None):
    h, w = im.shape[:2]

    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    warped_corners = H @ corners
    warped_corners = warped_corners / warped_corners[2, :]
    
    min_x = int(np.floor(np.min(warped_corners[0, :])))
    max_x = int(np.ceil(np.max(warped_corners[0, :])))
    min_y = int(np.floor(np.min(warped_corners[1, :])))
    max_y = int(np.ceil(np.max(warped_corners[1, :])))
    
    if output_shape is not None:
        out_h, out_w = output_shape
    else:
        out_h = max_y - min_y
        out_w = max_x - min_x

    if len(im.shape) == 3:
        warped = np.zeros((out_h, out_w, im.shape[2]), dtype=im.dtype)
    else:
        warped = np.zeros((out_h, out_w), dtype=im.dtype)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)

    H_inv = np.linalg.inv(H)
    for y_out in range(out_h):
        for x_out in range(out_w):
            x_world = x_out + min_x
            y_world = y_out + min_y
            p_out = np.array([x_world, y_world, 1.0])
            p_src = H_inv @ p_out
            
            if p_src[2] != 0:
                p_src = p_src / p_src[2]
            else:
                continue
            
            src_x = int(round(p_src[0]))
            src_y = int(round(p_src[1]))
            
            if 0 <= src_x < w and 0 <= src_y < h:
                warped[y_out, x_out] = im[src_y, src_x]
                mask[y_out, x_out] = 255
    
    return warped, mask

def warpImageBilinear(im, H, output_shape=None):
    h, w = im.shape[:2]

    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T
    
    warped_corners = H @ corners
    warped_corners = warped_corners / warped_corners[2, :]
    
    min_x = np.floor(np.min(warped_corners[0, :])).astype(int)
    max_x = np.ceil(np.max(warped_corners[0, :])).astype(int)
    min_y = np.floor(np.min(warped_corners[1, :])).astype(int)
    max_y = np.ceil(np.max(warped_corners[1, :])).astype(int)
    
    if output_shape is not None:
        out_h, out_w = output_shape
        min_x_used = 0
        min_y_used = 0
    else:
        out_h = max_y - min_y
        out_w = max_x - min_x
        min_x_used = min_x
        min_y_used = min_y

    if len(im.shape) == 3:
        warped = np.zeros((out_h, out_w, im.shape[2]), dtype=np.float32)
    else:
        warped = np.zeros((out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)

    H_inv = np.linalg.inv(H)
    
    for y_out in range(out_h):
        for x_out in range(out_w):
            x_out_world = x_out + min_x_used
            y_out_world = y_out + min_y_used
            p_out = np.array([x_out_world, y_out_world, 1.0])
            p_src = H_inv @ p_out
            
            if p_src[2] != 0:
                p_src = p_src / p_src[2]
            else:
                continue
            
            src_x = p_src[0]
            src_y = p_src[1]
            x0 = int(np.floor(src_x))
            x1 = x0 + 1
            y0 = int(np.floor(src_y))
            y1 = y0 + 1
            
            if x0 >= 0 and x1 < w and y0 >= 0 and y1 < h:
                wx1 = src_x - x0
                wx0 = 1.0 - wx1
                wy1 = src_y - y0
                wy0 = 1.0 - wy1
                
                if len(im.shape) == 3:
                    for c in range(im.shape[2]):
                        Ia = im[y0, x0, c]
                        Ib = im[y1, x0, c]
                        Ic = im[y0, x1, c]
                        Id = im[y1, x1, c]
                        
                        warped[y_out, x_out, c] = (
                            wx0 * wy0 * Ia +
                            wx0 * wy1 * Ib +
                            wx1 * wy0 * Ic +
                            wx1 * wy1 * Id
                        )
                else:
                    Ia = im[y0, x0]
                    Ib = im[y1, x0]
                    Ic = im[y0, x1]
                    Id = im[y1, x1]
                    
                    warped[y_out, x_out] = (
                        wx0 * wy0 * Ia +
                        wx0 * wy1 * Ib +
                        wx1 * wy0 * Ic +
                        wx1 * wy1 * Id
                    )
                
                mask[y_out, x_out] = 255
    
    warped = np.clip(warped, 0, 255).astype(im.dtype)
    
    return warped, mask

def rectify_image_example():

    im = cv2.imread('/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj3/media/IMG_6456.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    print(f"Image shape: {im.shape}")
    
    im1_pts = np.array([
        [1966, 2605],
        [5342, 1052],
        [6472, 3776],
        [3200, 4421]
    ], dtype=np.float32)
    
    im2_pts = np.array([
        [0, 0],
        [800, 0],
        [800, 600],
        [0, 600]
    ], dtype=np.float32)

    print("Source points (im1_pts):")
    print(im1_pts)
    print("\nTarget points (im2_pts):")
    print(im2_pts)

    H = computeH(im1_pts, im2_pts)
    warped_nn, mask_nn = warpImageNearestNeighbor(im, H)
    warped_bil, mask_bil = warpImageBilinear(im, H)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(im)
    pts_plot = np.vstack([im1_pts, im1_pts[0]])
    axes[0].plot(pts_plot[:, 0], pts_plot[:, 1], 'r-', linewidth=3)
    axes[0].plot(im1_pts[:, 0], im1_pts[:, 1], 'ro', markersize=10)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(warped_nn)
    axes[1].set_title('Nearest Neighbor (Custom)')
    axes[1].axis('off')
    
    axes[2].imshow(warped_bil)
    axes[2].set_title('Bilinear (Custom)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('rectification_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return warped_nn, warped_bil, mask_nn, mask_bil

def test_canvas_placement(img1, img2, H_1to2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.array([[0, 0, 1], [w1, 0, 1], [w1, h1, 1], [0, h1, 1]]).T
    warped_corners1 = H_1to2 @ corners1
    warped_corners1 = warped_corners1 / warped_corners1[2, :]
    
    corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
    
    all_x = np.concatenate([warped_corners1[0, :], corners2[:, 0]])
    all_y = np.concatenate([warped_corners1[1, :], corners2[:, 1]])
    
    min_x = int(np.floor(all_x.min()))
    max_x = int(np.ceil(all_x.max()))
    min_y = int(np.floor(all_y.min()))
    max_y = int(np.ceil(all_y.max()))
    
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    offset_x = -min_x
    offset_y = -min_y
    
    print(f"Canvas: {canvas_w} x {canvas_h}")
    print(f"Offset: ({offset_x}, {offset_y})")

    canvas_img2 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_img1 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_both = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    img2_x = offset_x
    img2_y = offset_y
    canvas_img2[img2_y:img2_y+h2, img2_x:img2_x+w2] = img2
    canvas_both[img2_y:img2_y+h2, img2_x:img2_x+w2] = img2
    
    print(f"img2 placed at: ({img2_x}, {img2_y}) to ({img2_x+w2}, {img2_y+h2})")

    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
    H_combined = T @ H_1to2
    
    warped1, mask1 = warpImageBilinear(img1, H_combined, output_shape=(canvas_h, canvas_w))
    canvas_img1 = warped1.copy()
    mask1_bool = mask1 > 0
    canvas_both[mask1_bool] = warped1[mask1_bool]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes[0, 0].imshow(canvas_img2)
    axes[0, 0].set_title('Canvas with img2 only (reference position)', fontsize=14)
    axes[0, 0].axhline(offset_y, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(offset_y + h2, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(offset_x, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(offset_x + w2, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(canvas_img1)
    axes[0, 1].set_title('Canvas with warped img1 only', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(canvas_both)
    axes[1, 0].set_title('Both images (img1 overwrites img2)', fontsize=14)
    axes[1, 0].axis('off')

    overlap = (mask1 > 0) & (canvas_img2.sum(axis=2) > 0)
    axes[1, 1].imshow(overlap, cmap='gray')
    axes[1, 1].set_title('Overlap region (white)', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    #plt.savefig('canvas_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nWarped img1 shape: {warped1.shape}")
    print(f"Valid pixels in img1: {np.sum(mask1 > 0)}")
    print(f"Overlap pixels: {np.sum(overlap)}")
    
    return canvas_both

def blend_two_images(img1, img2, H_1to2, accumulator=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    corners1 = np.array([[0, 0, 1], [w1, 0, 1], [w1, h1, 1], [0, h1, 1]]).T
    warped_corners1 = H_1to2 @ corners1
    warped_corners1 = warped_corners1 / warped_corners1[2, :]
    
    if accumulator is None:
        corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        all_x = np.concatenate([warped_corners1[0, :], corners2[:, 0]])
        all_y = np.concatenate([warped_corners1[1, :], corners2[:, 1]])
        
        min_x = int(np.floor(all_x.min()))
        max_x = int(np.ceil(all_x.max()))
        min_y = int(np.floor(all_y.min()))
        max_y = int(np.ceil(all_y.max()))
        
        canvas_w = max_x - min_x
        canvas_h = max_y - min_y
        offset_x = -min_x
        offset_y = -min_y
        
        weighted_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
        total_weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        
        img2_x = offset_x
        img2_y = offset_y
        mask2 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        mask2[img2_y:img2_y+h2, img2_x:img2_x+w2] = 255
        weight2 = distance_transform_edt(mask2 > 0)
        
        weighted_sum[img2_y:img2_y+h2, img2_x:img2_x+w2] = (
            img2.astype(float) * weight2[img2_y:img2_y+h2, img2_x:img2_x+w2][:, :, np.newaxis]
        )
        total_weight += weight2
        
    else:
        old_h, old_w = accumulator['sum'].shape[:2]
        old_min_x = accumulator['min_x']
        old_min_y = accumulator['min_y']
        old_max_x = old_min_x + old_w
        old_max_y = old_min_y + old_h
        new_min_x = warped_corners1[0, :].min()
        new_max_x = warped_corners1[0, :].max()
        new_min_y = warped_corners1[1, :].min()
        new_max_y = warped_corners1[1, :].max()
        min_x = int(np.floor(min(old_min_x, new_min_x)))
        max_x = int(np.ceil(max(old_max_x, new_max_x)))
        min_y = int(np.floor(min(old_min_y, new_min_y)))
        max_y = int(np.ceil(max(old_max_y, new_max_y)))
        
        canvas_w = max_x - min_x    
        canvas_h = max_y - min_y
        offset_x = -min_x
        offset_y = -min_y
        
        weighted_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
        total_weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)
        copy_x = old_min_x - min_x
        copy_y = old_min_y - min_y
        
        weighted_sum[copy_y:copy_y+old_h, copy_x:copy_x+old_w] = accumulator['sum']
        total_weight[copy_y:copy_y+old_h, copy_x:copy_x+old_w] = accumulator['weight']
    
    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
    H_combined = T @ H_1to2
    warped1, mask1 = warpImageBilinear(img1, H_combined, output_shape=(canvas_h, canvas_w))
    
    weight1 = distance_transform_edt(mask1 > 0)
    weighted_sum += warped1.astype(float) * weight1[:, :, np.newaxis]
    total_weight += weight1
    
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    has_weight = total_weight > 0
    result[has_weight] = (
        weighted_sum[has_weight] / total_weight[has_weight, np.newaxis]
    ).astype(np.uint8)
    
    accumulator = {
        'sum': weighted_sum,
        'weight': total_weight,
        'min_x': min_x,
        'min_y': min_y
    }
    
    return result, accumulator

def create_panorama_sequential(images, homographies):
    n = len(images)
    ref_idx = n // 2
    
    print(f"Using image {ref_idx} as reference (total: {n} images)")
    
    panorama = images[ref_idx].copy()
    accumulator = None

    for i in range(ref_idx - 1, -1, -1):
        print(f"Adding image {i} to panorama...")
        
        H_composed = np.eye(3)
        for j in range(i, ref_idx):
            H_composed = homographies[j] @ H_composed
        
        panorama, accumulator = blend_two_images(images[i], panorama, H_composed, accumulator)
        print(f"  Panorama size: {panorama.shape}")
        if accumulator:
            print(f"  World bounds: x=[{accumulator['min_x']}, {accumulator['min_x']+panorama.shape[1]}], " +
                  f"y=[{accumulator['min_y']}, {accumulator['min_y']+panorama.shape[0]}]")

    for i in range(ref_idx + 1, n):
        print(f"Adding image {i} to panorama...")
        
        H_composed = np.eye(3)
        for j in range(i - 1, ref_idx - 1, -1):
            H_composed = H_composed @ np.linalg.inv(homographies[j])
        
        panorama, accumulator = blend_two_images(images[i], panorama, H_composed, accumulator)
        print(f"  Panorama size: {panorama.shape}")
        if accumulator:
            print(f"  World bounds: x=[{accumulator['min_x']}, {accumulator['min_x']+panorama.shape[1]}], " +
                  f"y=[{accumulator['min_y']}, {accumulator['min_y']+panorama.shape[0]}]")
    
    return panorama

def load_panorama_data(base_path, image_numbers):

    images = []
    homographies = []

    for num in image_numbers:
        img_path = os.path.join(base_path, f'DSC_0{num}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        print(f"Loaded DSC_0{num}.jpg: {img.shape}")
    
    for i in range(len(image_numbers) - 1):
        num1 = image_numbers[i]
        num2 = image_numbers[i + 1]
        
        json_path = os.path.join(base_path, f'DSC_0{num1}_DSC_0{num2}.json')
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        im1_pts = np.array(data['im1Points'], dtype=np.float32)
        im2_pts = np.array(data['im2Points'], dtype=np.float32)
        
        H = computeH(im1_pts, im2_pts)
        homographies.append(H)
        
        print(f"Computed H: DSC_0{num1} -> DSC_0{num2}")
    
    print(f"\nTotal: {len(images)} images, {len(homographies)} homographies")
    
    return images, homographies

def create_panorama_from_data(base_path, image_numbers, output_name='panorama'):
    images, homographies = load_panorama_data(base_path, image_numbers)

    print("CREATING PANORAMA")
    
    panorama = create_panorama_sequential(images, homographies)

    plt.figure(figsize=(24, 8))
    plt.imshow(panorama)
    plt.axis('off')
    plt.title(f'Panorama: DSC_0{image_numbers[0]} to DSC_0{image_numbers[-1]}')
    plt.tight_layout()
    
    output_path = os.path.join(base_path, f'{output_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved panorama to: {output_path}")
    plt.show()
    
    return panorama