import os, sys, numpy as np, torch
import nibabel as nb
from tqdm import tqdm

sys.path.append('/path/to/project_root')
from autoencoderkl import AutoencoderKL
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42)

# ==================== Configuration Parameters ====================
vae_ckpt_path = '/path/to/project_root/ckpt/autoencoder_minloss.pth'
ckpt_dir = '/path/to/project_root/ckpt/ckpt_diffusion_stage1'
unet_ckpt_path = os.path.join(ckpt_dir, 'best_stage1.pth')

# Input/Output directories
input_dir = '/path/to/project_root/input_data/original/'
output_dir = '/path/to/project_root/output_data/'

# Create output subdirectories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'vae'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'diff'), exist_ok=True)

# Load scheduler config
import json
with open(os.path.join(ckpt_dir, 'scheduler_config.json'), 'r') as f:
    config = json.load(f)

print(f"Loaded config: beta=[{config.get('beta_start', 0.0001)}, {config.get('beta_end', 0.0195)}]")

# ==================== Utility Functions ====================
def block_ind(mask, sz_block=64):
    tmp = np.nonzero(mask)
    if len(tmp[0]) == 0: 
        return np.array([]), []
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
    vol = np.zeros(mask.shape)
    cnt = np.zeros(mask.shape)
    for i in range(inds.shape[0]):
        idx = inds[i]
        vol[idx[0]:idx[1]+1, idx[2]:idx[3]+1, idx[4]:idx[5]+1] += blocks[i]
        cnt[idx[0]:idx[1]+1, idx[2]:idx[3]+1, idx[4]:idx[5]+1] += 1.
    cnt[cnt < 0.5] = 1.
    return (vol / cnt) * mask

# ==================== Load Models ====================
autoencoder = AutoencoderKL(
    spatial_dims=3, in_channels=1, out_channels=1, 
    num_channels=(64, 128, 192), latent_channels=3, 
    num_res_blocks=1, norm_num_groups=16,
    attention_levels=(False, False, True)
).to(device)
autoencoder.load_state_dict(torch.load(vae_ckpt_path, map_location=device)['net'])
autoencoder.half().eval()
print("VAE loaded")

unet = DiffusionModelUNet(
    spatial_dims=3, in_channels=1, out_channels=1, 
    num_res_blocks=2, num_channels=(64, 128, 256), 
    attention_levels=(False, True, True),
    num_head_channels=(0, 128, 256)
).to(device)
ckpt = torch.load(unet_ckpt_path, map_location=device)
unet.load_state_dict(ckpt['unet'])
unet.eval()
print(f"UNet loaded from epoch {ckpt.get('epoch', '?')}")

scheduler = DDPMScheduler(
    num_train_timesteps=config.get('num_train_timesteps', 1000), 
    schedule="scaled_linear_beta",
    beta_start=config.get('beta_start', 0.0001), 
    beta_end=config.get('beta_end', 0.0195), 
    clip_sample=False
)

# ==================== Processing Functions ====================
@torch.no_grad()
def process_single(img_path, autoencoder, unet, scheduler, device):
    nii = nb.load(img_path)
    data = nii.get_fdata()
    affine = nii.affine
    mask = data > 0
    
    # Normalization
    if mask.sum() > 0:
        d_mean = np.mean(data[mask])
        d_std = max(np.std(data[mask]), 1e-8)
        data_norm = np.zeros_like(data)
        data_norm[mask] = (data[mask] - d_mean) / d_std
    else:
        d_mean, d_std = 0, 1
        data_norm = data
    
    print(f"Mean: {d_mean:.4f}, Std: {d_std:.4f}")
    print(f"Normalized - Mean: {np.mean(data_norm[mask]):.4f}, Std: {np.std(data_norm[mask]):.4f}, "
          f"Max: {np.max(data_norm[mask]):.4f}, Min: {np.min(data_norm[mask]):.4f}")
    
    inds, _ = block_ind(mask, sz_block=64)
    if len(inds) == 0:
        return None, None, None
    
    blocks_vae = np.zeros((inds.shape[0], 64, 64, 64))
    blocks_diff = np.zeros((inds.shape[0], 64, 64, 64))
    
    # Process block by block
    for j in range(inds.shape[0]):
        idx = inds[j]
        patch = data_norm[idx[0]:idx[1]+1, idx[2]:idx[3]+1, idx[4]:idx[5]+1]
        
        # Ensure patch size is 64x64x64
        if patch.shape != (64, 64, 64):
            temp = np.zeros((64, 64, 64))
            temp[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = temp
        
        patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).half().to(device)
        
        # VAE Reconstruction
        with torch.amp.autocast('cuda'):
            vae_out, _, _ = autoencoder(patch_t)
        blocks_vae[j] = vae_out.float().cpu().numpy().reshape(64, 64, 64)
        
        print(f"VAE output - Mean: {np.mean(blocks_vae[j]):.4f}, Std: {np.std(blocks_vae[j]):.4f}, "
              f"Max: {np.max(blocks_vae[j]):.4f}, Min: {np.min(blocks_vae[j]):.4f}")
        
        # Standard Diffusion Denoising (Stage 1: starting from pure noise)
        scheduler.set_timesteps(config.get('num_train_timesteps', 1000))
        sample = torch.randn_like(patch_t)
        
        with torch.amp.autocast('cuda'):
            for ts in scheduler.timesteps:
                noise_pred = unet(sample, torch.tensor([ts], device=device).long())
                step_out = scheduler.step(noise_pred, ts, sample)
                sample = step_out.prev_sample if hasattr(step_out, 'prev_sample') else step_out[0]
        
        blocks_diff[j] = sample.float().cpu().numpy().reshape(64, 64, 64)
        
        print(f"Diff output - Mean: {np.mean(blocks_diff[j]):.4f}, Std: {np.std(blocks_diff[j]):.4f}, "
              f"Max: {np.max(blocks_diff[j]):.4f}, Min: {np.min(blocks_diff[j]):.4f}")
    
    # Combine blocks
    vol_vae = block2brain(blocks_vae, inds, mask)
    vol_diff = block2brain(blocks_diff, inds, mask)
    
    # Denormalization
    vol_vae_denorm = np.zeros_like(vol_vae)
    vol_diff_denorm = np.zeros_like(vol_diff)
    vol_vae_denorm[mask] = vol_vae[mask] * d_std + d_mean
    vol_diff_denorm[mask] = vol_diff[mask] * d_std + d_mean
    
    return vol_vae_denorm, vol_diff_denorm, affine

# ==================== Main Processing Pipeline ====================
def list_nifti_files(directory):
    """List all .nii.gz and .nii files in directory"""
    files = []
    for f in os.listdir(directory):
        if f.endswith('.nii.gz') or f.endswith('.nii'):
            files.append(f)
    return sorted(files)

# Get all input files
all_files = list_nifti_files(input_dir)
print(f"Found {len(all_files)} files in {input_dir}")

# Process one by one
for filename in tqdm(all_files):
    img_path = os.path.join(input_dir, filename)
    
    # Get base filename (remove .nii.gz or .nii)
    if filename.endswith('.nii.gz'):
        basename = filename.replace('.nii.gz', '')
    else:
        basename = filename.replace('.nii', '')
    
    try:
        print(f"\nProcessing: {filename}")
        
        vol_vae, vol_diff, affine = process_single(
            img_path, autoencoder, unet, scheduler, device
        )
        
        if vol_vae is None:
            print(f"Skipped {filename} (empty mask)")
            continue
        
        # Save VAE result
        vae_output_path = os.path.join(output_dir, 'vae', f'{basename}.nii.gz')
        nb.save(nb.Nifti1Image(vol_vae.astype(np.float32), affine), vae_output_path)
        
        # Save Diffusion result
        diff_output_path = os.path.join(output_dir, 'diff', f'{basename}.nii.gz')
        nb.save(nb.Nifti1Image(vol_diff.astype(np.float32), affine), diff_output_path)
        
        print(f"✅ Saved: {basename}.nii.gz")
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print(f"✅ All done! Results saved to:")
print(f"  VAE:  {os.path.join(output_dir, 'vae')}")
print(f"  Diff: {os.path.join(output_dir, 'diff')}")
print(f"{'='*60}")