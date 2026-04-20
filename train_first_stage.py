import os, sys, numpy as np, torch
import json, random, logging
import matplotlib.pyplot as plt
import nibabel as nb
from torch.nn import L1Loss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vaedataset import UKBDataset
from autoencoderkl import AutoencoderKL
from perceptual import PerceptualLoss
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# ==================== Path Configuration ====================
ckpt_dir = '/path/to/your/vae_ckpt_dir'
os.makedirs(ckpt_dir, exist_ok=True)
train_json_path = '/path/to/your/train_AE.json'
val_json_path = '/path/to/your/val_AE.json'

# ==================== Training Configuration ====================
batch_size, n_epochs, MinValLoss = 16, 300, 10000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ckpt_dir, 'training_log.txt')),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==================== Utility Functions ====================
def block_ind(mask, sz_block=64, sz_pad=0):

    tmp = np.nonzero(mask)
    if len(tmp[0]) == 0: return np.array([]), []
    
    xmin, xmax = np.min(tmp[0]), np.max(tmp[0])
    ymin, ymax = np.min(tmp[1]), np.max(tmp[1])
    zmin, zmax = np.min(tmp[2]), np.max(tmp[2])
    
    nx = int(np.ceil((xmax - xmin + 1) / sz_block)) + sz_pad
    ny = int(np.ceil((ymax - ymin + 1) / sz_block)) + sz_pad
    nz = int(np.ceil((zmax - zmin + 1) / sz_block)) + sz_pad
    
    xind_block = np.round(np.linspace(xmin, xmax - sz_block + 1, nx)).astype(int)
    yind_block = np.round(np.linspace(ymin, ymax - sz_block + 1, ny)).astype(int)
    zind_block = np.round(np.linspace(zmin, zmax - sz_block + 1, nz)).astype(int)
    
    ind_block = []
    for x in xind_block:
        for y in yind_block:
            for z in zind_block:
                ind_block.append([x, x+sz_block-1, y, y+sz_block-1, z, z+sz_block-1])
    return np.array(ind_block), [xmin, xmax, ymin, ymax, zmin, zmax]

def block2brain(blocks, inds, mask):

    vol_brain = np.zeros(mask.shape)
    vol_count = np.zeros(mask.shape)
    for i in range(inds.shape[0]):
        idx = inds[i]
        vol_brain[idx[0]:idx[1]+1, idx[2]:idx[3]+1, idx[4]:idx[5]+1] += blocks[i]
        vol_count[idx[0]:idx[1]+1, idx[2]:idx[3]+1, idx[4]:idx[5]+1] += 1.
    
    vol_count[vol_count < 0.5] = 1.
    return (vol_brain / vol_count) * mask


# ==================== Visualization Function (Test Logic) ====================
def visualize_validation(model, json_path, epoch, save_dir):
    """
    Randomly sample cases every 10 epochs for whole-brain reconstruction and visualization
    """
    try:
        # Use Pandas to read, as it automatically handles different JSON formats (records, columns, index, etc.)
        df = pd.read_json(json_path)
        if 'high_path' not in df.columns:
            logging.error(f"'high_path' column not found in JSON file. Existing columns: {df.columns}")
            return
        # Get deduplicated list of paths
        all_paths = df['high_path'].unique().tolist()
        
    except Exception as e:
        logging.error(f"Failed to read JSON file: {e}")
        return
        
    num_samples = 5
    if len(all_paths) < num_samples:
        samples = all_paths
    else:
        samples = random.sample(all_paths, num_samples)
    
    real_count = len(samples)
    if real_count == 0:
        logging.warning("No valid validation samples found, skipping visualization.")
        return
        
    fig, axes = plt.subplots(real_count, 6, figsize=(18, 3 * real_count))
    # If there is only one row, axes will be a 1D array. Force it to 2D for unified indexing axes[i, k]
    if real_count == 1:
        axes = axes.reshape(1, -1)
        
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Get device
    device = next(model.parameters()).device
    model.eval()
    logging.info(f"Generating visualization for Epoch {epoch} with {real_count} samples...")

    # Process samples one by one
    for i, img_path in enumerate(samples):
        try:
            if not os.path.exists(img_path):
                logging.warning(f"File does not exist: {img_path}, skipping.")
                continue
                
            # --- Load Image ---
            nii = nb.load(img_path)
            data = nii.get_fdata()
            mask = data > 0
            if mask.sum() > 0:
                d_mean, d_std = np.mean(data[mask]), np.std(data[mask])
                # Avoid numerical explosion caused by too small variance
                if d_std < 1e-8: d_std = 1.0 
                data_norm = np.zeros_like(data)
                data_norm[mask] = (data[mask] - d_mean) / d_std
            else:
                data_norm = data # All black
                
            inds, _ = block_ind(mask, sz_block=64)
            
            if len(inds) == 0: 
                logging.warning(f"No valid brain block found for sample {os.path.basename(img_path)}")
                continue
            
            blocks_recon = np.zeros((inds.shape[0], 64, 64, 64))
            
            # --- Model Inference ---
            with torch.no_grad():
                for j in range(inds.shape[0]):
                    idx = inds[j]
                    # Safe slicing
                    patch = data_norm[idx[0]:idx[1]+1, idx[2]:idx[3]+1, idx[4]:idx[5]+1]
                    
                    # Edge case protection: if the sliced block is not 64x64x64
                    if patch.shape != (64, 64, 64):
                        temp_patch = np.zeros((64, 64, 64))
                        dx, dy, dz = patch.shape
                        temp_patch[:dx, :dy, :dz] = patch
                        patch = temp_patch
                        
                    patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    
                    # Model forward pass
                    # Assuming the model returns (recon, mu, logvar)
                    output = model(patch_t)
                    if isinstance(output, tuple):
                        recon = output[0]
                    else:
                        recon = output
                    
                    blocks_recon[j] = recon.cpu().numpy().reshape(64, 64, 64)

            vol_recon = block2brain(blocks_recon, inds, mask)
            cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
            
            # Limit coordinate range to prevent out-of-bounds
            cx = min(cx, data.shape[0]-1)
            cy = min(cy, data.shape[1]-1)
            cz = min(cz, data.shape[2]-1)
            
            slices = [
                data_norm[cx, :, :], data_norm[:, cy, :], data_norm[:, :, cz], # Original
                vol_recon[cx, :, :], vol_recon[:, cy, :], vol_recon[:, :, cz]   # Reconstruction
            ]
            titles = ['Org Ax', 'Org Cor', 'Org Sag', 'Rec Ax', 'Rec Cor', 'Rec Sag']
            
            for k, slc in enumerate(slices):
                ax = axes[i, k]
                # Rotating 90 degrees is usually to fit the visual habit of medical images
                ax.imshow(np.rot90(slc), cmap='gray', vmin=-3, vmax=3) 
                if i == 0: ax.set_title(titles[k])
                ax.axis('off')
                
        except Exception as e:
            logging.error(f"Error visualizing {img_path}: {e}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    save_path = os.path.join(save_dir, f'viz_epoch_{epoch:04d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved visualization to {save_path}")

# ==================== Training Preparation ====================
# Use locally defined Dataset
dataset_train = UKBDataset(json_path=train_json_path)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=True)

dataset_val = UKBDataset(json_path=val_json_path)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True, pin_memory=True)

# Model
autoencoder = AutoencoderKL(spatial_dims=3, in_channels=1, out_channels=1, num_channels=(64, 128, 192), latent_channels=3, num_res_blocks=1, norm_num_groups=16, attention_levels=(False,False,True)).to(device)

# Loss and Optimization
l1_loss = L1Loss()
loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).to(device)
KL_loss = lambda z_mu, z_sigma: (0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])).sum() / z_mu.shape[0]

perceptual_weight, kl_weight = 0.001, 1e-6
optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=n_epochs)

# ==================== Main Training Loop ====================
logging.info(f"Start Training on {device}... Data size: {len(dataset_train)} patches")
log = np.zeros([n_epochs, 8])

for epoch in tqdm(range(1, n_epochs + 1)):
    autoencoder.train()
    epoch_recon_loss_list = []

    # Train Loop
    for batch, data in enumerate(loader_train, 1):
        images = data['high'].to(device)
        optimizer_g.zero_grad(set_to_none=True) 
        
        reconstruction, z_mu, z_sigma = autoencoder(images)
        
        loss_l1 = l1_loss(reconstruction.float(), images.float())
        loss_p = loss_perceptual(reconstruction.float(), images.float())
        loss_k = KL_loss(z_mu, z_sigma)
        
        loss_g = loss_l1 + kl_weight * loss_k + perceptual_weight * loss_p

        epoch_recon_loss_list.append(loss_l1.item())
        loss_g.backward()
        optimizer_g.step()

    train_loss_avg = np.mean(epoch_recon_loss_list)
    scheduler.step()
    log[epoch-1, 0] = train_loss_avg

    # Validation Loop (Calc Loss)
    if epoch % 1 == 0:
        autoencoder.eval()
        val_loss_list = []
        with torch.no_grad():
            for data in loader_val:
                images = data['high'].to(device)
                reconstruction, z_mu, z_sigma = autoencoder(images) 
                val_loss_list.append(l1_loss(reconstruction.float(), images.float()).item())
        
        val_loss_avg = np.mean(val_loss_list)
        log[epoch-1, 3] = val_loss_avg
        logging.info(f"EPOCH {epoch:04d} | Train L1: {train_loss_avg:.5f} | Val L1: {val_loss_avg:.5f}")

        # Save Best
        if val_loss_avg < MinValLoss:
            MinValLoss = val_loss_avg
            torch.save({'net': autoencoder.state_dict(), 'epoch': epoch}, f"{ckpt_dir}/autoencoder_minloss.pth")
            logging.info(f"Saved MinLoss Model: {MinValLoss:.5f}")

    # Save Checkpoint & Log
    np.save(os.path.join(ckpt_dir, 'log2.npy'), log)
    state = {'net': autoencoder.state_dict(), 'optim': optimizer_g.state_dict(), 'epoch': epoch}
    torch.save(state, f"{ckpt_dir}/autoencoder_eachepoch.pth")

    # Visualization (Every 10 epochs)
    if epoch % 10 == 0:
        visualize_validation(autoencoder, val_json_path, epoch, ckpt_dir)