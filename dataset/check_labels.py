# Quick check script - downloads segmentation, checks for label 0, analyzes unique labels
# Does NOT save to disk

from caveclient import CAVEclient
import imageryclient as ic
import numpy as np

print("Connecting to MICrONS dataset...")
datastack_name = 'minnie65_public'
client = CAVEclient(datastack_name)
client.version = 1412

print("Initializing ImageryClient...")
img_client = ic.ImageryClient(client=client)

# Download larger chunk
print("\nDownloading 128×128×128 segmentation volume...")
ctr = [240640, 207872, 21360]  # (x, y, z) in voxels
dimSize = 128
bbox_size = (dimSize, dimSize, dimSize)

_, seg_3d = img_client.image_and_segmentation_cutout(
    ctr,
    bbox_size=bbox_size,
    split_segmentations=False,
    scale_to_bounds=True
)

print(f"✓ Downloaded shape: {seg_3d.shape}")
print(f"✓ Data type: {seg_3d.dtype}")
print(f"✓ Total voxels: {seg_3d.size:,}")

# Check for 0
print("\n" + "="*60)
if 0 in seg_3d:
    print("✓ FOUND: Label 0 is present (likely background/empty space)")
    zero_count = np.sum(seg_3d == 0)
    print(f"  → {zero_count:,} voxels with label 0 ({zero_count/seg_3d.size*100:.2f}%)")
else:
    print("✗ NOT FOUND: No label 0 in this volume")
print("="*60)

# Show top labels by frequency
print("\nTop 10 most frequent labels:")
unique_labels, counts = np.unique(seg_3d, return_counts=True)
sorted_by_freq = sorted(zip(unique_labels, counts), key=lambda x: -x[1])

for i, (label, count) in enumerate(sorted_by_freq[:10], 1):
    percent = count/seg_3d.size*100
    print(f"  {i:2d}. Label {label:18d}: {count:8,} voxels ({percent:6.2f}%)")

print(f"\nTotal unique labels: {len(unique_labels):,}")
print(f"Min label: {unique_labels[0]}")
print(f"Max label: {unique_labels[-1]}")

print("\n✓ Done (no files saved)")
