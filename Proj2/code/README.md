# Project 2: Fun with Filters and Frequencies

This folder contains implementations for CS180 Project 2, covering convolution operations, edge detection, image sharpening, hybrid images, and multiresolution blending.

## Project Structure

This project is organized into four main parts, each corresponding to a specific Python file:

- **Part 1**: `part1.py` - Convolution operations and edge detection
- **Part 2.1**: `part2.py` - Image sharpening using unsharp masking
- **Part 2.2**: `hybrid_image_starter.py` - Hybrid image creation
- **Part 2.3/2.4**: `blend.py` - Gaussian/Laplacian stacks and multiresolution blending

## File Descriptions
x
### `part1.py` - Part 1: Fun with Filters
**Core convolution and edge detection implementation**

**Main Functions:**
- `convolution_four_loops(image, kernel)` - Naive 4-nested-loop implementation
- `convolution_two_loops(image, kernel)` - Optimized 2-loop version
- `finite_difference(image_path)` - Basic edge detection using Dx, Dy operators
- `gaussian(image_path, sigma, kernel_size)` - DoG-based edge detection

**Key Parameters:**
- `sigma` (float): Gaussian blur strength (default: 1.7)
- `kernel_size` (int): Filter size (default: 15)
- Edge threshold percentile for binarization (default: 91st percentile)

---

### `part2.py` - Part 2.1: Image "Sharpening"
**Unsharp masking implementation for image sharpening**

**Main Functions:**
- `unsharp_masking(image, sigma, alpha)` - Multi-step sharpening process
- `unsharp_masking_single_conv(image, sigma, alpha)` - Single convolution approach
- `blur_and_resharpen_test(image, blur_sigma, sharp_sigma, alpha)` - Evaluation method
- `process_image_sharpening(image_path)` - Complete pipeline with visualization

**Key Parameters:**
- `sigma` (float): Gaussian blur amount for unsharp mask (default: 0.6)
- `alpha` (float): Sharpening strength multiplier (range: 0.5-2.0)
- `alpha_values`: List of alpha values for comparison `[0.5, 1.0, 1.5, 2.0]`

---

### `hybrid_image_starter.py` - Part 2.2: Hybrid Images
**Creates hybrid images using frequency domain filtering**

**Main Functions:**
- `hybrid_image(im1, im2, sigma1, sigma2)` - Core hybrid creation
- `frequency_analysis(im1, im2, hybrid, sigma1, sigma2)` - FFT visualization
- `pyramids(image, N)` - Multi-scale pyramid generation

**Key Parameters:**
- `sigma1` (float): High-pass cutoff frequency (smaller = more high freq details)
- `sigma2` (float): Low-pass cutoff frequency (larger = more low freq content)
- `N` (int): Number of pyramid levels (default: 5)

---

### `blend.py` - Part 2.3/2.4: Multiresolution Blending
**Gaussian/Laplacian stacks and seamless image blending**

**Main Functions:**
- `create_gaussian_stack(image, num_levels, sigma_base)` - Multi-scale Gaussian filtering
- `create_laplacian_stack(gaussian_stack)` - Band-pass decomposition
- `blend_images_with_mask(img1, img2, mask, num_levels)` - Complete blending pipeline
- `reconstruct_from_laplacian(laplacian_stack)` - Image reconstruction

**Key Parameters:**
- `num_levels` (int): Number of frequency bands (default: 6)
- `sigma_base` (float): Base Gaussian sigma (default: 1.0, scales as 2^level)
- `mask`: Binary/grayscale blending mask (same size as input images)

---

## Running the Code

### Part 1: Convolution and Edge Detection
```python
# Convolution implementations
test_convolutions()

# Basic edge detection
finite_difference('/path/to/image.jpg')

# DoG-based edge detection
gaussian('/path/to/image.jpg', sigma=1.7, kernel_size=15)
```

### Part 2.1: Image Sharpening
```python
# Process image with multiple alpha values
results = process_image_sharpening('/path/to/image.jpg')

# Test blur-and-resharpen
blurred, resharpened = blur_and_resharpen_test(image, blur_sigma=2.0, sharp_sigma=1.0, alpha=1.2)
```

### Part 2.2: Hybrid Images
```python
# Align images first
im1_aligned, im2_aligned = align_images(image1, image2)

# Create hybrid with different sigma values
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1=10.0, sigma2=8.0)

# Analyze frequency content
frequency_analysis(im1_aligned, im2_aligned, hybrid, sigma1, sigma2)
```

### Part 2.3/2.4: Multiresolution Blending
```python
# Create stacks
gaussian_stack = create_gaussian_stack(image, num_levels=6)
laplacian_stack = create_laplacian_stack(gaussian_stack)

# Blend images
blended = blend_images_with_mask(img1, img2, mask, num_levels=6)

# Classic oraple example
demonstrate_oraple_process() 
```