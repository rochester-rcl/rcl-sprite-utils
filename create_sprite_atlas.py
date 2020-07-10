import cv2
import os
from natsort import natsorted
import math
import numpy as np
from functools import reduce


def get_tile_size(impath):
    im = cv2.imread(impath)
    h, w, channels = im.shape
    if h != w:
        raise ValueError("Input Image Shape Must be Square")
    return h


def get_frame_count(animations):
    cnt = 0
    for key in animations:
        cnt += len(animations[key])
    return cnt


def pack_atlas(atlas, animations, tile_size):
    max_pos = atlas.shape[0]
    metadata = {
        'info': {
            'tile_size': tile_size
        }
    }
    x = 0
    y = 0
    for key in animations:
        frames = animations[key]
        for frame in frames:
            image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
            atlas[y:y+tile_size, x:x+tile_size] = image
            if x == max_pos - tile_size:
                x = 0
                y += tile_size
            else:
                x += tile_size
    cv2.imwrite('test.png', atlas)


def compile_texture_atlas(animations):
    key = next(iter(animations))
    tile_size = get_tile_size(animations[key][0])
    n_frames = get_frame_count(animations)
    dim = math.ceil(math.sqrt(n_frames))
    imsize = int(math.pow(2, math.ceil(math.log(dim * tile_size) / math.log(2))))
    atlas = np.zeros((imsize, imsize, 4), dtype=np.uint8)
    packed, metadata = pack_atlas(atlas, animations, tile_size)


def read_animations(directory):
    animations = {}
    for dirname, dirnames, files in os.walk(directory):
        # figure out what directory the files are in, add to sprite sheet
        for f in files:
            name = os.path.basename(dirname)
            if name != dirname:
                fpath = os.path.join(os.path.abspath(dirname), f)
                if name not in animations:
                    animations[name] = [fpath]
                else:
                    animations[name].append(fpath)
    for key in animations:
        animations[key] = natsorted(animations[key])
    return animations


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Create a sprite atlas from segmented sprites")
    parser.add_argument(
        "-d", "--directory", help="The root directory to search for sprite folders", required=True)
    parser.add_argument(
        "-o", "--output", help="The output folder to save the atlas", required=True)
    args = vars(parser.parse_args())
    animations = read_animations(args["directory"])
    compile_texture_atlas(animations)
