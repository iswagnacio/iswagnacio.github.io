from feature_detect import test_harris_anms, extract_feature_descriptors, match_image_pair
from weighted_avg import computeH, test_canvas_placement, rectify_image_example, create_panorama_from_data, load_panorama_data
from laplacian import create_panorama_from_data_pyramid
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    #warped_nn, warped_bil, mask_nn, mask_bil = rectify_image_example()
    base_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj3/media/mosaic_1'
    image_numbers = [305,306,307,308,309,310]

    # Weighted Average Blending
    panorama = create_panorama_from_data(base_path, image_numbers)

    # Laplacian Pyramid Blending
    #panorama = create_panorama_from_data_pyramid(base_path, image_numbers, output_name='panorama_0865_to_0866_pyramid')

    '''image_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj3/media/mosaic_1/DSC_0305.jpg'
    coords_all, coords_anms, h, img_gray, img_rgb = test_harris_anms(
        image_path, 
        num_features=500,
        output_dir=base_path
    )'''
    '''coords, descriptors = detect_and_extract_features(
        f'{base_path}/DSC_0305.jpg',
        num_features=500,
        output_dir=base_path
    )
    matches, coords1, coords2, desc1, desc2 = match_image_pair(
        f'{base_path}/DSC_0305.jpg',
        f'{base_path}/DSC_0306.jpg',
        num_features=500,
        ratio_threshold=0.8,
        output_dir=base_path
    )'''