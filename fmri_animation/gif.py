import os
import re
import numpy as np
import imageio.v3 as iio
from glob import glob


def extract_integer(filename, pattern):
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else float('inf')

def sort_files(file_list, pattern):
    return sorted(file_list, key=lambda x: extract_integer(x, pattern))

def load_image_files(file_pattern):
    image_files = glob(file_pattern)
    image_files = sort_files(image_files, file_pattern.replace(r'*', r'(\d+)'))
    return image_files

def make_gif_from_images(image_files, out_path, fps=300, delete_files=True):
    frames = np.stack([iio.imread(f) for f in image_files], axis=0)
    iio.imwrite(out_path, frames, format='gif', fps=fps)
    if delete_files:
        for f in image_files:
            os.remove(f)