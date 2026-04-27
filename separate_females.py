import os
import pandas as pd
import shutil


def separate_females(attr_path, src_img_dir, dest_img_dir):
    # 1. Setup directories
    if not os.path.exists(dest_img_dir):
        os.makedirs(dest_img_dir)
        print(f"Created directory: {dest_img_dir}")

    # 2. Parse the attribute file
    # CelebA format:
    # Line 1: Number of images
    # Line 2: Attribute names
    print("Reading attribute file...")
    with open(attr_path, 'r') as f:
        lines = f.readlines()
        attr_names = lines[1].strip().split()

    # Load into DataFrame (skip first 2 rows)
    df = pd.read_csv(attr_path, sep=r'\s+', skiprows=2, names=['image_id'] + attr_names, engine='python')

    # 3. Filter for Females
    # In CelebA, Male=1 and Female=-1
    females_df = df[df['Male'] == -1]
    female_filenames = females_df['image_id'].tolist()
    total_females = len(female_filenames)

    print(f"Found {total_females} female images in attributes.")

    # 4. Copying process
    count = 0
    for filename in female_filenames:
        src_path = os.path.join(src_img_dir, filename)
        dest_path = os.path.join(dest_img_dir, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            count += 1
            if count % 1000 == 0:
                print(f"Copied {count}/{total_females}...")
        else:
            # Skip if the image isn't in the source folder (for partial datasets)
            continue

    print(f"\nDone! Successfully copied {count} photos to {dest_img_dir}.")


if __name__ == "__main__":
    # Update these paths to match your local setup
    separate_females(
        attr_path='list_attr_celeba.txt',
        src_img_dir='img_celeba',
        dest_img_dir='females_uncropped'
    )