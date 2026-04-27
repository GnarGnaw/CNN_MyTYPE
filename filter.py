import os

# Filenames
ATTR_FILE = 'attr.txt'  # Your reference file (118165 entries)
NEW_LAND_FILE = 'list_landmarks_celeba.txt'  # The source file (202599 entries)
OUTPUT_FILE = 'landmarks_cropped.txt'  # The filtered result


def filter_landmarks():
    # 1. Get the list of filenames we want to keep
    print(f"Reading {ATTR_FILE}...")
    valid_filenames = set()
    with open(ATTR_FILE, 'r') as f:
        # Skip the first two lines (count and header)
        lines = f.readlines()[2:]
        for line in lines:
            parts = line.split()
            if parts:
                valid_filenames.add(parts[0])

    print(f"Found {len(valid_filenames)} target images.")

    # 2. Filter the new landmark file
    print(f"Filtering {NEW_LAND_FILE}...")
    filtered_rows = []
    header = ""

    with open(NEW_LAND_FILE, 'r') as f:
        # Save the total count line and header line
        f.readline()
        header = f.readline()

        for line in f:
            parts = line.split()
            if not parts:
                continue

            filename = parts[0]
            if filename in valid_filenames:
                filtered_rows.append(line)

    # 3. Write the new file
    print(f"Writing {len(filtered_rows)} entries to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"{len(filtered_rows)}\n")
        f.write(header)
        f.writelines(filtered_rows)

    print("Done! Use this new 'landmarks.txt' for training.")


if __name__ == "__main__":
    filter_landmarks()