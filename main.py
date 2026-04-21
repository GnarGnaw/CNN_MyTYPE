import os
import shutil

source_dir = 'img_align_celeba'
target_dir = 'females'
attr_file = 'list_attr_celeba.txt'
landmarks_file = 'list_landmarks_celeba.txt'

new_attr_file = 'attr.txt'
new_landmarks_file = 'landmarks.txt'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

female_filenames = set()

with open(attr_file, 'r') as f, open(new_attr_file, 'w') as out_f:
    lines = f.readlines()
    header_count = lines[0].strip()
    attr_names = lines[1].strip()

    male_index = attr_names.split().index('Male')

    filtered_lines = []
    for line in lines[2:]:
        parts = line.split()
        if parts[male_index + 1] == '-1':
            filtered_lines.append(line)
            female_filenames.add(parts[0])

    out_f.write(f"{len(filtered_lines)}\n")
    out_f.write(f"{attr_names}\n")
    out_f.writelines(filtered_lines)

with open(landmarks_file, 'r') as f, open(new_landmarks_file, 'w') as out_f:
    lines = f.readlines()
    header_names = lines[1].strip()

    filtered_lines = []
    for line in lines[2:]:
        filename = line.split()[0]
        if filename in female_filenames:
            filtered_lines.append(line)

    out_f.write(f"{len(filtered_lines)}\n")
    out_f.write(f"{header_names}\n")
    out_f.writelines(filtered_lines)

for filename in female_filenames:
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(target_dir, filename)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)