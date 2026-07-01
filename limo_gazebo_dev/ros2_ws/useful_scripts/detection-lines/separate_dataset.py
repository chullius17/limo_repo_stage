import os
import random
import shutil

# --- CONFIGURATION ---
# Path to the directory where Roboflow dumped everything
base_dir = ""
train_img_dir = os.path.join(base_dir, "train", "images")
train_lbl_dir = os.path.join(base_dir, "train", "labels")

# Target directories for validation and test splits
splits = {
    "valid": (os.path.join(base_dir, "valid", "images"), os.path.join(base_dir, "valid", "labels")),
    "test": (os.path.join(base_dir, "test", "images"), os.path.join(base_dir, "test", "labels"))
}

# Target percentages (20% validation, 10% test)
valid_ratio = 0.20
test_ratio = 0.10

# --- MAIN SCRIPT ---
# Create target directories if they don't exist
for img_path, lbl_path in splits.values():
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(lbl_path, exist_ok=True)

# Get all images currently in the train folder
images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle images randomly to ensure an unbiased split
random.seed(42)  # Fixed seed for reproducibility
random.shuffle(images)

num_images = len(images)
num_valid = int(num_images * valid_ratio)
num_test = int(num_images * test_ratio)

# Determine which images go where
valid_images = images[:num_valid]
test_images = images[num_valid:num_valid + num_test]

def move_files(image_list, target_img_dir, target_lbl_dir):
    """Moves images and their corresponding labels to the target directories."""
    for img_name in image_list:
        # Move image
        src_img = os.path.join(train_img_dir, img_name)
        dst_img = os.path.join(target_img_dir, img_name)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        
        # Move corresponding label file (.txt)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_lbl = os.path.join(train_lbl_dir, label_name)
        dst_lbl = os.path.join(target_lbl_dir, label_name)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)

# Execute the move
print(f"Total images found in train: {num_images}")
print(f"Moving {len(valid_images)} images to validation set...")
move_files(valid_images, splits["valid"][0], splits["valid"][1])

print(f"Moving {len(test_images)} images to test set...")
move_files(test_images, splits["test"][0], splits["test"][1])

print("Dataset manually split with success! Remaining images are kept in train.")