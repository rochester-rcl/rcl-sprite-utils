import cv2
import numpy as np
import math


def segment(impath, outpath, **kwargs):
    im = cv2.imread(impath)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, 0)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:
            continue
        boxes.append(cv2.boundingRect(contour))
    pad_len = len(str(len(boxes)))
    idx = 0
    # sort boxes by min value
    boxes = sorted(boxes, key=lambda b: b[0])
    max_square_bbox = get_max_bbox(boxes)
    for box in boxes:
        x, y, w, h = box
        outfile = "{}/{}_{num:0{width}}.png".format(outpath, kwargs['prefix'], num=idx, width=pad_len)
        roi = im[y:y+h, x:x+w]
        cv2.imwrite(outfile, fit_to_box(max_square_bbox, roi, **kwargs))
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1)
        idx += 1
    # TODO reconstitute all boxes as a max_square_bbox * len(boxes) spritesheet for animation
    cv2.imwrite("{}/all.jpg".format(outpath), im)


def get_max_bbox(boxes):
    max_bbox = sorted(boxes, key=lambda b: b[1]*b[2])[0]
    max_side = max((max_bbox[1], max_bbox[2]))
    pow_2 = math.pow(2, math.ceil(math.log(max_side) / math.log(2)))
    return [0, 0, pow_2, pow_2]


def fit_to_box(box, roi, **kwargs):
    h, w, channels = roi.shape
    x, y, out_h, out_w = box
    delta_w = out_w - w
    delta_h = out_h - h
    top = int(delta_h / 2)
    bottom = int(delta_h - top)
    left = int(delta_w / 2)
    right = int(delta_w - left)
    if "background" in kwargs:
        low = np.array(kwargs["background"], np.uint8)
        high = np.array([val + 50 for val in kwargs["background"]], np.uint8)
        mask = cv2.inRange(roi, low, high)
        out_image = cv2.cvtColor(roi, cv2.COLOR_RGB2RGBA)
        out_image[mask != 0] = [0, 0, 0, 0]
        return cv2.copyMakeBorder(out_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
    return cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="segment a sprite sheet")
    parser.add_argument("-i", "--input", help="The input file", required=True)
    parser.add_argument("-o", "--output", help="The output folder to save segmented images", required=True)
    parser.add_argument("-b", "--background", help="The background color to convert to an alpha value. In the format R, G, B", required=False)
    parser.add_argument("-p", "--prefix", help="The prefix to add to the individual sprite files", default="sprite")
    args = vars(parser.parse_args())

    if args["background"]:
        bg = [int(val) for val in args["background"].split(",")]
        segment(args["input"], args["output"], background=bg, prefix=args["prefix"])
    else:
        segment(args["input"], args["output"])
