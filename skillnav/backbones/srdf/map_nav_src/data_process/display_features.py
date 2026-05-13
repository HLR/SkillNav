import h5py
import os
from tqdm import tqdm


hdf5_path = '../datasets/R2R/features/clip_vit-b16_mp3d_hm3d_gibson.hdf5'
target_key = '00102-r77mpaAYUEc_00002'

with h5py.File(hdf5_path, 'r') as f:
    all_keys = list(f.keys())
    print(f"Total keys: {len(all_keys)}")

    print("\nFirst 10 keys:")
    for k in all_keys[:10]:
        print(k)

    print("\nLast 10 keys:")
    for k in all_keys[-10:]:
        print(k)

    if target_key in f:
        print(f"\n✅ Key '{target_key}' found!")
        data = f[target_key][()]
        print(f"Data shape: {data.shape}")
    else:
        print(f"\n❌ Key '{target_key}' NOT found.")
