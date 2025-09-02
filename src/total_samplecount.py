import os

# Path to your dataset root
dataset_path = "D:\College\Second Year\EDI\Pneumonia_Detection\DATA"

# Splits and classes
splits = ['train', 'val', 'test']
classes = ['NORMAL', 'PNEUMONIA']

# Dictionary to store counts
image_counts = {split: {cls: 0 for cls in classes} for split in splits}

# Count images
for split in splits:
    for cls in classes:
        class_path = os.path.join(dataset_path, split, cls)
        if os.path.exists(class_path):
            image_counts[split][cls] = len([
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ])

# Print results
total_images = 0
for split in splits:
    split_total = sum(image_counts[split].values())
    total_images += split_total
    print(f"\n{split.upper()} SET:")
    for cls in classes:
        print(f"  {cls}: {image_counts[split][cls]} images")
    print(f"  Total: {split_total} images")

print(f"\n🔢 Combined Total: {total_images} images")