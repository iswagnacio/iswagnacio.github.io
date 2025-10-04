from util import computeH, test_canvas_placement, rectify_image_example, create_panorama_from_data
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    warped_nn, warped_bil, mask_nn, mask_bil = rectify_image_example()
    '''base_path = '/Users/junwei/Fall2025/CS180/iswagnacio.github.io/Proj3/media/mosaic_1'
    image_numbers = [305,306,307]
    
    panorama = create_panorama_from_data(base_path, image_numbers, output_name='panorama_0305_to_0307')'''