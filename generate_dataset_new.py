import numpy as np
import pandas as pd
import nibabel as nb
import copy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def normalize_image(img, mask):

    valid_pixels = img[mask > 0.5]
    if len(valid_pixels) == 0:
        return 0.0, 1.0
    img_mean = np.mean(valid_pixels)
    img_std = np.std(valid_pixels)
    return img_mean, img_std

def block_ind(mask, sz_block=64, sz_pad=0):
    # find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask)
    if len(tmp[0]) == 0: # Handle empty mask case
        return np.array([]), []
        
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    zmin = np.min(zind); zmax = np.max(zind)
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]
    
    # calculate number of blocks along each dimension
    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    zlen = zmax - zmin + 1
    nx = int(np.ceil(xlen / sz_block)) + sz_pad
    ny = int(np.ceil(ylen / sz_block)) + sz_pad
    nz = int(np.ceil(zlen / sz_block)) + sz_pad
    
    # determine starting and ending indices of each block
    xstart = xmin
    ystart = ymin
    zstart = zmin
    xend = xmax - sz_block + 1
    yend = ymax - sz_block + 1
    zend = zmax - sz_block + 1
    
    # Handle case where dimension is smaller than block size
    if xend < xstart: xend = xstart
    if yend < ystart: yend = ystart
    if zend < zstart: zend = zstart

    xind_block = np.round(np.linspace(xstart, xend, nx))
    yind_block = np.round(np.linspace(ystart, yend, ny))
    zind_block = np.round(np.linspace(zstart, zend, nz))
    
    ind_block = []
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
            for kk in np.arange(0, zind_block.shape[0]):
                ind_block.append([
                    xind_block[ii], xind_block[ii]+sz_block-1, 
                    yind_block[jj], yind_block[jj]+sz_block-1, 
                    zind_block[kk], zind_block[kk]+sz_block-1
                ])
    
    return np.array(ind_block).astype(int), ind_brain

def process_split(root_path, split_name):

    split_dir = os.path.join(root_path, split_name)
    if not os.path.exists(split_dir):
        print(f"path {split_dir} not exist")
        return
    
    files = [f for f in os.listdir(split_dir) if f.endswith('.nii.gz')]
    files.sort()
    
    data_list = []
    
    
    for filename in tqdm(files):
        img_path = os.path.join(split_dir, filename)
        
        if filename.endswith('.nii.gz'):
            fake_mask_name = filename.replace('.nii.gz', '_mask.nii.gz')
        else:
            fake_mask_name = filename + '_mask'
        fake_mask_path = os.path.join(split_dir, fake_mask_name)
        
        try:
            nii = nb.load(img_path)
            high = nii.get_fdata()
            
            mask = high > 0
            high_expand = np.expand_dims(high, -1)
            mask_expand = np.expand_dims(mask, -1)
            
            high_mean, high_std = normalize_image(high_expand, mask_expand)
            ind_block, _ = block_ind(mask, 64, 0)
            
            if len(ind_block) == 0:
                continue

            for i in range(ind_block.shape[0]):
                row = {
                    'index': str(len(data_list)), 
                    'high_path': img_path,
                    'seg_path': fake_mask_path, 
                    'mean_high': high_mean,
                    'std_high': high_std,
                    'shape0': int(ind_block[i, 0]),
                    'shape1': int(ind_block[i, 1]),
                    'shape2': int(ind_block[i, 2]),
                    'shape3': int(ind_block[i, 3]),
                    'shape4': int(ind_block[i, 4]),
                    'shape5': int(ind_block[i, 5])
                }
                data_list.append(row)
                
        except Exception as e:
            continue
            
    df = pd.DataFrame(data_list)
    output_json = f'/{split_name}.json'
    df.to_json(output_json)


def main():
    dpRoot = '/data/path/to/your/project'
    splits = ['train', 'val', 'test']
    
    for split in splits:
        process_split(dpRoot, split)

if __name__ == "__main__":
    main()
