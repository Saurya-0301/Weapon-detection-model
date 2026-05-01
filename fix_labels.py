import os

label_dir = "weapon_dataset/train/labels"

for file in os.listdir(label_dir):

    path = os.path.join(label_dir, file)

    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:

        parts = line.split()

        cls = int(parts[0])

        # remove invalid class
        if cls > 3:
            continue

        new_lines.append(line)

    with open(path, "w") as f:
        f.writelines(new_lines)

print("Label cleanup complete.")