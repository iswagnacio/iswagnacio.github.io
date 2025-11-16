import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim=2, max_freq_log2=10):
        super().__init__()
        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = max_freq_log2 + 1
        self.output_dim = input_dim + 2 * self.num_freqs * input_dim
        
    def forward(self, x):
        encoded = [x]
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        return torch.cat(encoded, dim=-1)


class NeuralField2D(nn.Module):
    def __init__(self, max_freq_log2=10, hidden_dim=256, num_layers=3):
        super().__init__()
        self.pe = PositionalEncoding(input_dim=2, max_freq_log2=max_freq_log2)
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.pe.output_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, coords):
        encoded = self.pe(coords)
        rgb = self.mlp(encoded)
        return rgb


class ImageDataloader:
    def __init__(self, image, batch_size=10000, device='cpu'):
        self.H, self.W, _ = image.shape
        self.batch_size = batch_size
        self.device = device

        self.image = torch.from_numpy(image).float() / 255.0
        self.image = self.image.to(device)

        y_coords = torch.arange(self.H, dtype=torch.float32) / self.H
        x_coords = torch.arange(self.W, dtype=torch.float32) / self.W
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        self.coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H*W, 2)
        self.colors = self.image.reshape(-1, 3)  # (H*W, 3)
        self.coords = self.coords.to(device)
        self.colors = self.colors.to(device)
        
        self.num_pixels = self.H * self.W
        
    def sample_batch(self):
        indices = torch.randint(0, self.num_pixels, (self.batch_size,), device=self.device)
        return self.coords[indices], self.colors[indices]
    
    def get_all(self):
        return self.coords, self.colors


def compute_psnr(mse):
    return 10.0 * torch.log10(1.0 / mse)


def train_neural_field(image, max_freq_log2=10, hidden_dim=256, num_layers=3,
                      learning_rate=1e-2, num_iterations=3000, batch_size=10000,
                      device='cpu', save_dir='part1_results'):
    os.makedirs(save_dir, exist_ok=True)

    dataloader = ImageDataloader(image, batch_size=batch_size, device=device)
    H, W = dataloader.H, dataloader.W
    model = NeuralField2D(
        max_freq_log2=max_freq_log2,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    psnr_history = []
    progression_iterations = [0, 500, 1500, num_iterations - 1]
    
    print(f"\nTraining for {num_iterations} iterations")
    pbar = tqdm(range(num_iterations))
    for iteration in pbar:
        coords_batch, colors_batch = dataloader.sample_batch()
        
        # Forward pass
        pred_colors = model(coords_batch)
        loss = criterion(pred_colors, colors_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute PSNR
        psnr = compute_psnr(loss)
        psnr_history.append(psnr.item())
        
        # Update progress
        pbar.set_description(f"Loss: {loss.item():.6f} | PSNR: {psnr.item():.2f} dB")
        
        if iteration in progression_iterations:
            with torch.no_grad():
                coords_all, _ = dataloader.get_all()
                pred_colors_all = model(coords_all)
                pred_image = pred_colors_all.reshape(H, W, 3).cpu().numpy()

                pred_image_uint8 = (pred_image * 255).astype(np.uint8)
                Image.fromarray(pred_image_uint8).save(
                    os.path.join(save_dir, f'iter_{iteration:04d}.png')
                )
    
    return model, psnr_history


def reconstruct_full_image(model, H, W, device='cpu'):
    model.eval()
    y_coords = torch.arange(H, dtype=torch.float32) / H
    x_coords = torch.arange(W, dtype=torch.float32) / W
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)

    batch_size = 10000
    all_colors = []
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i+batch_size]
            batch_colors = model(batch_coords)
            all_colors.append(batch_colors)

    all_colors = torch.cat(all_colors, dim=0)
    image = all_colors.reshape(H, W, 3).cpu().numpy()
    
    return image


def plot_training_progression(image_dir, iterations, output_file):
    n = len(iterations)
    if n <= 5:
        rows, cols = 1, n
    elif n <= 10:
        rows, cols = 2, (n + 1) // 2
    else:
        rows, cols = (n + 4) // 5, 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = axes.reshape(rows, cols)

    for idx, iteration in enumerate(iterations):
        row = idx // cols
        col = idx % cols
        
        img_path = os.path.join(image_dir, f'iter_{iteration:04d}.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            if rows == 1:
                axes[col].imshow(img)
                axes[col].set_title(f'Iteration {iteration}')
                axes[col].axis('off')
            else:
                axes[row][col].imshow(img)
                axes[row][col].set_title(f'Iteration {iteration}')
                axes[row][col].axis('off')

    total_subplots = rows * cols
    for idx in range(n, total_subplots):
        row = idx // cols
        col = idx % cols
        if rows == 1:
            axes[col].remove()
        else:
            axes[row][col].remove()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_psnr_curve(psnr_history, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(psnr_history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Training Progress: PSNR over Iterations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using Mac")
    else:
        device = 'cpu'
        print("Using CPU")

    os.makedirs('part1_deliverables', exist_ok=True)

    #Deliverable 1
    fox_image_path = 'Proj4/Media/part1_2.png'
    
    if os.path.exists(fox_image_path):
        fox_image = np.array(Image.open(fox_image_path).convert('RGB'))
        model_fox, psnr_fox = train_neural_field(
            fox_image,
            max_freq_log2=10,
            hidden_dim=256,
            num_layers=3,
            learning_rate=1e-2,
            num_iterations=3000,
            batch_size=10000,
            device=device,
            save_dir='part1_deliverables/fox_progression'
        )
        
        create_training_progression(
            'part1_deliverables/fox_progression',
            'part1_deliverables/fox_training_progression.png',
            title="Training Progression - Fox Image"
        )
    else:
        print(f"Image not found at {fox_image_path}")
    
    #Deliverable 2:
    custom_image_path = 'Media\part1_2.png' 
    if os.path.exists(custom_image_path):
        custom_image = np.array(Image.open(custom_image_path).convert('RGB'))

        model_custom, psnr_custom = train_neural_field(
            custom_image,
            max_freq_log2=10,
            hidden_dim=256,
            num_layers=3,
            learning_rate=1e-2,
            num_iterations=3000,
            batch_size=10000,
            device=device,
            save_dir='part1_deliverables/custom_progression'
        )

        create_training_progression(
            'part1_deliverables/custom_progression',
            'part1_deliverables/custom_training_progression.png',
            title="Training Progression - Custom Image"
        )
        deliverable_image = custom_image
        deliverable_psnr = psnr_custom

    elif os.path.exists(fox_image_path):
        deliverable_image = fox_image
        deliverable_psnr = psnr_fox
    else:
        print("No images found.")
        return

    #Deliverable 3: Hyperparameter Grid (2x2)
    create_hyperparameter_grid(deliverable_image, device)

    #Deliverable 4: PSNR Curve
    plot_psnr_curve(deliverable_psnr, 'part1_deliverables/psnr_curve.png')


def create_training_progression(image_dir, output_file, title="Training Progression"):
    progression_files = [
        (0, 'iter_0000.png'),
        (500, 'iter_0500.png'),
        (1500, 'iter_1500.png'),
        (2999, 'iter_2999.png')
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, (iteration, filename) in enumerate(progression_files):
        img_path = os.path.join(image_dir, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f'Iteration {iteration}', fontsize=12)
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, f'Missing\nIter {iteration}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_hyperparameter_grid(image, device):
    experiments = [
        (4, 128, "Low Freq (L=4)\nSmall Width (128)"),
        (4, 256, "Low Freq (L=4)\nLarge Width (256)"),
        (10, 128, "High Freq (L=10)\nSmall Width (128)"),
        (10, 256, "High Freq (L=10)\nLarge Width (256)"),
    ]
    
    results = []
    
    for i, (max_freq, hidden_dim, label) in enumerate(experiments):
        model, psnr_history = train_neural_field(
            image,
            max_freq_log2=max_freq,
            hidden_dim=hidden_dim,
            num_layers=3,
            learning_rate=1e-2,
            num_iterations=2000, 
            batch_size=10000,
            device=device,
            save_dir=f'part1_deliverables/grid_temp_{i}'
        )

        final_image = reconstruct_full_image(model, image.shape[0], image.shape[1], device)
        final_psnr = psnr_history[-1]
        results.append((final_image, label, final_psnr))
        import shutil
        shutil.rmtree(f'part1_deliverables/grid_temp_{i}', ignore_errors=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, (result_image, label, psnr) in enumerate(results):
        row, col = i // 2, i % 2
        display_image = (result_image * 255).astype(np.uint8)
        axes[row, col].imshow(display_image)
        axes[row, col].set_title(f"{label}\nPSNR: {psnr:.1f} dB", fontsize=11)
        axes[row, col].axis('off')
    
    plt.suptitle('Hyperparameter Comparison: PE Frequency vs Network Width', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('part1_deliverables/hyperparameter_grid_2x2.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()