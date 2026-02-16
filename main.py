import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np

# Mocking the imports based on the project structure
from src.quantizer import DeadZoneLDZCompander
from src.utils_quantization import attach_weight_quantizers, toggle_quantization

def main():
    print("Setting up ResNet18 example with DeadZone quantization...")

    model = resnet18(num_classes=10)

    EXCLUDE = ['conv1', 'bn1', 'downsample', 'fc']
    QUANT_ARGS = {
        'fixed_bit_val': 4,
        'max_bits': 8,
        'init_bit_logit': 1.0,     # Reduced slightly to show more effect
        'init_deadzone_logit': 1.0, # Reduced to show some initial sparsity
        'learnable_bit': True,
        'learnable_deadzone': True
    }

    # Attach weight quantizers
    # Note: User referred to DeadZoneLDZ, using DeadZoneLDZCompander from src/quantizer.py
    attach_weight_quantizers(
        model=model,
        exclude_layers=EXCLUDE,
        quantizer=DeadZoneLDZCompander,
        quantizer_kwargs=QUANT_ARGS,
        enabled=True
    )

    # 3. Separate params into groups for optimized learning rates
    # Quantization parameters (logits) often require different learning rates than weights
    base_params = []
    dz_params = []
    bit_params = []

    for name, param in model.named_parameters():
        if 'logit_dz' in name:
            dz_params.append(param)
        elif 'logit_bit' in name:
            bit_params.append(param)
        else:
            base_params.append(param)

    lr = 1e-4
    lr_dz = 0.5
    lr_bit = 1e-2
    weight_decay = 1e-4
    weight_decay_dz = 0.4
    weight_decay_bit = 0.0

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': lr,          'weight_decay': weight_decay},
        {'params': dz_params,   'lr': lr_dz,       'weight_decay': weight_decay_dz},
        {'params': bit_params,  'lr': lr_bit,      'weight_decay': weight_decay_bit},
    ])

    toggle_quantization(model, enabled=True)
    
    print("Starting training loop (simulation with FakeData)...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    dataset = torchvision.datasets.FakeData(
        size=500, 
        image_size=(3, 32, 32), # ResNet18 is often used with 224, but 32 works for CIFAR-style/standard resnet
        num_classes=10, 
        transform=transforms.ToTensor()
    ) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print("Training finished.")

    # 5. Visualize the sparsity of one of the layers
    model.eval()
    # Correcting target_layer: 'layer1.0.conv1' was excluded via EXCLUDE list.
    # 'layer1.0.conv2' is quantized.
    target_layer = 'layer1.0.conv2' # A standard 3x3 convolution
    print(f"\nVisualizing sparsity for layer: {target_layer}")
    
    # Extract the layer
    modules = dict(model.named_modules())
    if target_layer in modules:
        layer = modules[target_layer]
        
        # The weight attribute is now a parametrized tensor
        with torch.no_grad():
            # Ensure we get the quantized weight, not the latent weight
            # The parametrized module automatically returns quantized weight on access if enabled
            w_quant = layer.weight
            weights_np = w_quant.detach().cpu().numpy()
            
            # weights_np shape: (out_channels, in_channels, k, k)
            out_ch, in_ch, k, k_w = weights_np.shape
            
            total = weights_np.size
            zeros = np.sum(weights_np == 0)
            sparsity = (zeros / total) * 100
            
            print(f"Total parameters: {total}")
            print(f"Zero parameters: {zeros}")
            print(f"Sparsity: {sparsity:.2f}%")
            
            # Create a 2D Visualization
            # We will create a large grid image where each cell is a kernel (k x k_w)
            # We arrange them with Out Channels on Y-axis and In Channels on X-axis (or flattened)
            # User wants to see "different groups". 
            # In a standard conv layer, the "groups" usually refers to group convolution, but here 
            # groups=1 (standard). So we visualize the interactions between all In and Out channels.
            
            # Setup grid dimensions
            padding = 1
            grid_h = out_ch * (k + padding) + padding
            grid_w = in_ch * (k_w + padding) + padding
            
            # Initialize grid with a middle gray or separate color for borders if desired, 
            # but user asked for black/white. Let's make background gray to distinguish kernel bounds,
            # and map 0->Black, Non-Zero->White.
            grid_image = np.ones((grid_h, grid_w)) * 0.5 
            
            for o in range(out_ch):
                for i in range(in_ch):
                    kernel = weights_np[o, i, :, :]
                    
                    # Create binary mask: 0 (Black) if zero, 1 (White) if non-zero
                    # User request: "black or white dependent on if it is zero or non zero"
                    binary_kernel = (kernel != 0).astype(float)
                    
                    # Calculate position
                    row_start = padding + o * (k + padding)
                    col_start = padding + i * (k_w + padding)
                    
                    grid_image[row_start : row_start + k, col_start : col_start + k_w] = binary_kernel

            plt.figure(figsize=(12, 12))
            # Use 'gray' colormap: 0 is Black, 1 is White. 0.5 (padding) will be gray.
            plt.imshow(grid_image, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            plt.title(f"Sparsity Map - {target_layer}\n(Rows: Out Channels, Cols: In Channels)\nBlack=Zero, White=Non-Zero, Gray=Separator")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig("resnet_sparsity_map.png", dpi=300)
            print("Visualization saved as 'resnet_sparsity_map.png'")
    else:
        print(f"Layer {target_layer} not found in model.")

if __name__ == "__main__":
    main()
