import os, sys, numpy as np, torch, torch.nn.functional as F
import json, random, logging, matplotlib.pyplot as plt
import nibabel as nb, pandas as pd
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

sys.path.append('/path/to/project_root')
from autoencoderkl import AutoencoderKL
from vaedataset import UKBDataset
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
torch.manual_seed(42); np.random.seed(42)

# ==================== Path Configuration ====================
vae_ckpt_path = '/path/to/project_root/ckpt/autoencoder_minloss.pth'
ckpt_dir = '/path/to/project_root/ckpt/ckpt_diffusion_stage1'
os.makedirs(ckpt_dir, exist_ok=True)

train_json_path = '/path/to/project_root/src/json/train_AE.json'
val_json_path = '/path/to/project_root/src/json/val_AE.json'

# ==================== Training Configuration ====================
batch_size_per_gpu = 1
batch_size = batch_size_per_gpu * max(1, n_gpus)
sample_ratio = 0.5
n_epochs = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
    handlers=[logging.FileHandler(os.path.join(ckpt_dir, 'training_log.txt')), logging.StreamHandler(sys.stdout)])

# ==================== Utility Functions ====================

def block_ind(mask, sz_block=64):
    tmp = np.nonzero(mask)
    if len(tmp[0]) == 0: return np.array([]), []
    xmin, xmax = np.min(tmp[0]), np.max(tmp[0])
    ymin, ymax = np.min(tmp[1]), np.max(tmp[1])
    zmin, zmax = np.min(tmp[2]), np.max(tmp[2])
    nx = int(np.ceil((xmax - xmin + 1) / sz_block))
    ny = int(np.ceil((ymax - ymin + 1) / sz_block))
    nz = int(np.ceil((zmax - zmin + 1) / sz_block))
    xind = np.round(np.linspace(xmin, max(xmin, xmax - sz_block + 1), max(1, nx))).astype(int)
    yind = np.round(np.linspace(ymin, max(ymin, ymax - sz_block + 1), max(1, ny))).astype(int)
    zind = np.round(np.linspace(zmin, max(zmin, zmax - sz_block + 1), max(1, nz))).astype(int)
    ind_block = []
    for x in xind:
        for y in yind:
            for z in zind:
                ind_block.append([x, x+sz_block-1, y, y+sz_block-1, z, z+sz_block-1])
    return np.array(ind_block), [xmin, xmax, ymin, ymax, zmin, zmax]

def block2brain(blocks, inds, mask):
    vol = np.zeros(mask.shape); cnt = np.zeros(mask.shape)
    for i in range(inds.shape[0]):
        idx = inds[i]
        vol[idx[0]:idx[1]+1, idx[2]:idx[3]+1, idx[4]:idx[5]+1] += blocks[i]
        cnt[idx[0]:idx[1]+1, idx[2]:idx[3]+1, idx[4]:idx[5]+1] += 1.
    cnt[cnt < 0.5] = 1.
    return (vol / cnt) * mask

def get_model(model):
    return model.module if hasattr(model, 'module') else model

# ==================== Stage 1 Standard DDPM Training ====================

def train_stage1(unet, loader, optimizer, scaler, scheduler, device):
    """
    Stage 1 DDPM Training:
    - Standard forward diffusion process
    - Random timesteps
    """
    unet.train()
    losses = []
    
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    for batch in loader:
        images = batch['high'].to(device, non_blocking=True)
        b = images.shape[0]
        
        optimizer.zero_grad(set_to_none=True)
        
        # Random sample timesteps
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(images)
        
        # Add noise
        sqrt_alpha_t = sqrt_alphas_cumprod[timesteps].view(b, 1, 1, 1, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[timesteps].view(b, 1, 1, 1, 1)
        noisy = sqrt_alpha_t * images + sqrt_one_minus_alpha_t * noise
        
        # Predict noise
        with torch.amp.autocast('cuda'):
            noise_pred = unet(noisy, timesteps)
            loss = F.mse_loss(noise_pred.float(), noise.float())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    
    return np.mean(losses)

@torch.no_grad()
def validate_stage1(unet, loader, scheduler, device):
    unet.eval()
    losses = []
    
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    for batch in loader:
        images = batch['high'].to(device, non_blocking=True)
        b = images.shape[0]
        
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=device).long()
        noise = torch.randn_like(images)
        
        sqrt_alpha_t = sqrt_alphas_cumprod[timesteps].view(b, 1, 1, 1, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[timesteps].view(b, 1, 1, 1, 1)
        noisy = sqrt_alpha_t * images + sqrt_one_minus_alpha_t * noise
        
        with torch.amp.autocast('cuda'):
            noise_pred = unet(noisy, timesteps)
            loss = F.mse_loss(noise_pred.float(), noise.float())
            
        losses.append(loss.item())
    
    return np.mean(losses)

# ==================== Visualization (Inference) ====================

@torch.no_grad()
def visualize_pipeline(autoencoder, unet, scheduler, json_path, epoch, save_dir):
    try:
        df = pd.read_json(json_path)
        all_paths = df['high_path'].unique().tolist()
    except:
        return
    if len(all_paths) == 0:
        return
    
    img_path = random.choice(all_paths)
    if not os.path.exists(img_path):
        return
    logging.info(f"Viz: {os.path.basename(img_path)}")
    
    nii = nb.load(img_path)
    data = nii.get_fdata()
    mask = data > 0
    
    if mask.sum() > 0:
        d_mean, d_std = np.mean(data[mask]), max(np.std(data[mask]), 1e-8)
        data_norm = np.zeros_like(data)
        data_norm[mask] = (data[mask] - d_mean) / d_std
    else:
        data_norm = data
    
    inds, _ = block_ind(mask, sz_block=64)
    if len(inds) == 0:
        return
    
    autoencoder.eval()
    unet.eval()
    ae_model = get_model(autoencoder)
    unet_model = get_model(unet)
    
    blocks_vae = np.zeros((inds.shape[0], 64, 64, 64))
    blocks_diff = np.zeros((inds.shape[0], 64, 64, 64))
    
    # Pre-compute DDPM sampling coefficients
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    betas = scheduler.betas.to(device)
    alphas = 1 - betas
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
    posterior_log_variance = torch.log(posterior_variance)
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
    posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)
    
    for j in range(inds.shape[0]):
        idx = inds[j]
        patch = data_norm[idx[0]:idx[1]+1, idx[2]:idx[3]+1, idx[4]:idx[5]+1]
        if patch.shape != (64, 64, 64):
            temp = np.zeros((64, 64, 64))
            temp[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = temp
        
        patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # VAE Denoising
        with torch.amp.autocast('cuda'):
            vae_out, _, _ = ae_model(patch_t.half())
        vae_out = vae_out.float()
        blocks_vae[j] = vae_out.cpu().numpy().reshape(64, 64, 64)
        
        # DDPM Denoising: from T-1 to 0
        sample = torch.randn_like(patch_t)
        for t in reversed(range(0, scheduler.config.num_train_timesteps)):
            t_tensor = torch.tensor([t], device=device).long()
            
            with torch.amp.autocast('cuda'):
                noise_pred = unet_model(sample.half(), t_tensor)
            noise_pred = noise_pred.float()
            
            # Predict x_0
            x_recon = (sample - sqrt_one_minus_alphas_cumprod[t] * noise_pred) / sqrt_alphas_cumprod[t]
            x_recon = torch.clamp(x_recon, -3, 3)
            
            # posterior mean
            posterior_mean = posterior_mean_coef1[t] * x_recon + posterior_mean_coef2[t] * sample
            
            if t > 0:
                noise = torch.randn_like(sample)
                sample = posterior_mean + torch.exp(0.5 * posterior_log_variance[t]) * noise
            else:
                sample = posterior_mean
        
        blocks_diff[j] = sample.cpu().numpy().reshape(64, 64, 64)
    
    vol_vae = block2brain(blocks_vae, inds, mask)
    vol_diff = block2brain(blocks_diff, inds, mask)
    
    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
    view_names = ['Axial', 'Sagittal', 'Coronal']
    slices_orig = [data_norm[:, :, cz], data_norm[cx, :, :], data_norm[:, cy, :]]
    slices_vae = [vol_vae[:, :, cz], vol_vae[cx, :, :], vol_vae[:, cy, :]]
    slices_diff = [vol_diff[:, :, cz], vol_diff[cx, :, :], vol_diff[:, cy, :]]
    
    vmin, vmax = -2, 2
    for row, (view, s_orig, s_vae, s_diff) in enumerate(zip(view_names, slices_orig, slices_vae, slices_diff)):
        axes[row, 0].imshow(np.rot90(s_orig), cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 0].set_title('Original' if row == 0 else '')
        axes[row, 0].set_ylabel(view); axes[row, 0].axis('off')
        axes[row, 1].imshow(np.rot90(s_vae), cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title('VAE' if row == 0 else ''); axes[row, 1].axis('off')
        axes[row, 2].imshow(np.rot90(s_diff), cmap='gray', vmin=vmin, vmax=vmax)
        axes[row, 2].set_title('Diffusion' if row == 0 else ''); axes[row, 2].axis('off')
    
    fig.suptitle(f'Stage1 Epoch {epoch}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'viz_stage1_ep{epoch:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved viz for epoch {epoch}")

# ==================== Main Program ====================

num_workers = min(16, os.cpu_count())
dataset_train = UKBDataset(json_path=train_json_path)
dataset_val = UKBDataset(json_path=val_json_path)

n_samples_per_epoch = int(len(dataset_train) * sample_ratio)
train_sampler = RandomSampler(dataset_train, replacement=False, num_samples=n_samples_per_epoch)
loader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, 
                          drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# Load VAE (Frozen) - Used for Visualization comparison
autoencoder = AutoencoderKL(spatial_dims=3, in_channels=1, out_channels=1, num_channels=(64, 128, 192),
                            latent_channels=3, num_res_blocks=1, norm_num_groups=16,
                            attention_levels=(False, False, True)).to(device)
autoencoder.load_state_dict(torch.load(vae_ckpt_path, map_location=device)['net'])
autoencoder.half().eval()
for p in autoencoder.parameters():
    p.requires_grad = False
if n_gpus > 1:
    autoencoder = torch.nn.DataParallel(autoencoder)
logging.info(f"Loaded VAE from {vae_ckpt_path}")

# Create UNet
unet = DiffusionModelUNet(spatial_dims=3, in_channels=1, out_channels=1, num_res_blocks=2,
                          num_channels=(64, 128, 256), attention_levels=(False, True, True),
                          num_head_channels=(0, 128, 256)).to(device)

if n_gpus > 1:
    unet = torch.nn.DataParallel(unet)

# Scheduler
scheduler = DDPMScheduler(
    num_train_timesteps=1000, 
    schedule="scaled_linear_beta", 
    beta_start=0.0001,
    beta_end=0.0195,
    clip_sample=False,
)

config = {
    'num_train_timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.0195
}
with open(os.path.join(ckpt_dir, 'scheduler_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

logging.info("="*60)
logging.info("Stage 1: Standard DDPM Training")
logging.info(f"Epochs: {n_epochs}")
logging.info("="*60)

optimizer = torch.optim.Adam(get_model(unet).parameters(), lr=5e-5)
scaler = torch.amp.GradScaler('cuda')
lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

log = np.zeros([n_epochs, 3])  # [train_loss, val_loss, lr]
min_loss = float('inf')

logging.info(f"GPUs: {n_gpus}, Batch: {batch_size}")
logging.info(f"Data: {len(dataset_train)} total, {n_samples_per_epoch} per epoch")
logging.info("="*60)

# =============== Training Loop ===============
for epoch in tqdm(range(n_epochs), desc="Stage1"):
    train_loss = train_stage1(unet, loader_train, optimizer, scaler, scheduler, device)
    lr_sch.step()
    val_loss = validate_stage1(unet, loader_val, scheduler, device)
    
    log[epoch] = [train_loss, val_loss, optimizer.param_groups[0]['lr']]
    
    if (epoch + 1) % 5 == 0:
        logging.info(f"Epoch {epoch:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")
    
    if val_loss < min_loss:
        min_loss = val_loss
        torch.save({
            'unet': get_model(unet).state_dict(), 
            'epoch': epoch, 
            'val_loss': val_loss, 
            'config': config
        }, f"{ckpt_dir}/best_stage1.pth")
    
    np.save(os.path.join(ckpt_dir, 'log.npy'), log)
    torch.save({
        'unet': get_model(unet).state_dict(), 
        'optim': optimizer.state_dict(), 
        'epoch': epoch, 
        'config': config
    }, f"{ckpt_dir}/latest.pth")
    
    if (epoch + 1) % 10 == 0:
        visualize_pipeline(autoencoder, unet, scheduler, val_json_path, epoch, ckpt_dir)

# Final Save
torch.save({
    'unet': get_model(unet).state_dict(), 
    'epoch': n_epochs - 1, 
    'val_loss': min_loss, 
    'config': config
}, f"{ckpt_dir}/stage1_final.pth")

logging.info("\n" + "="*60)
logging.info(f"Stage 1 Done! Best val loss: {min_loss:.5f}")
logging.info("="*60)

# Plot Curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(log[:, 0], label='Train', alpha=0.7)
plt.plot(log[:, 1], label='Val', alpha=0.7)
plt.legend(); plt.title('Loss'); plt.xlabel('Epoch')
plt.subplot(1, 2, 2)
plt.plot(log[:, 2]); plt.yscale('log'); plt.title('LR'); plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig(os.path.join(ckpt_dir, 'curves.png'), dpi=150)
plt.close()

logging.info("Done!")