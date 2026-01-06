# file for downloading 3D segmentation data from MICrONS dataset
# Requires caveclient and imageryclient packages
# Run `pip install -r requirements.txt` to install dependencies

from caveclient import CAVEclient
import imageryclient as ic
import numpy as np
import os

datastack_name = 'minnie65_public'
client = CAVEclient(datastack_name)
client.version = 1412


# Connect to MICrONS dataset ---
datastack_name = 'minnie65_public'
client = CAVEclient(datastack_name)
client.version = 1412

# Initialize ImageryClient ---
img_client = ic.ImageryClient(client=client)

# Define a center coordinate in global MICrONS space ---
# You can get this from Neuroglancer or metadata
ctr = [240640, 207872, 21360]   # (x, y, z) in voxels

# Download a 3D cutout ---
# bbox_size: spatial extent in XY (pixels)
# z_height: number of slices in Z
bbox_size = (1024, 1024, 1024)

_, seg_3d = img_client.image_and_segmentation_cutout(
    ctr,
    bbox_size=bbox_size,
    split_segmentations=False,    # Return as single 3D label array
    scale_to_bounds=True
)

print("Segmentation shape:", seg_3d.shape)

# Save results
# Directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Output path in the same directory
out_path = os.path.join(script_dir, "microns_segmentation_3d.npy")

# Save next to where the script is
np.save(out_path, seg_3d)
print(f"Saved 3D segmentation as: {out_path}")
