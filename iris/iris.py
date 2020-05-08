import cv2 
import numpy as np
import sys
import os
import argparse

#Todo: delete after testing
import random


har_path = "./haarfiles"
face_cascade = cv2.CascadeClassifier(os.path.join(har_path,'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(har_path,'haarcascade_eye.xml'))


def detect_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return eyes

def crop_eyes(img, eyes):
    i_images = []
    for (x,y,w,h) in eyes:
        i_images.append(img[y:y+w, x:x+h, :])
    return i_images

def save_img_batch(output_path, output_images, filename_base="eye"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, img in enumerate(output_images):
        for j, eye in enumerate(img):
            filename = "img{!s}_{!s}{!s}.jpg".format(i, filename_base, j)
            cv2.imwrite(os.path.join(output_path, filename), eye)

def main(args):
    if args.cam_id:
        cap = cv2.VideoCapture(int(args.cam_id))
        output_images = []
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                i_images = crop_eyes(frame, detect_eyes(frame))
                output_images.append(i_images)
                if args.show:
                    for i, eye in enumerate(i_images):
                        cv2.imshow("eye-{!s}".format(i), eye)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                cv2.destroyAllWindows()
                print(len(output_images))
                save_img_batch(args.output, output_images)
                break
        return

    if not args.filename:
        raise ValueError("No filename specified")

    img = cv2.imread(args.filename)
    eyes = detect_eyes(img)
    i_images = crop_eyes(img, eyes)
    num = 1
    base, ext = os.path.splitext(args.filename)

    for i_img in i_images:
        cv2.imwrite(os.path.join(args.output, "{!s}_eye{!s}{!s}".format(base, num, ext)), i_img)
        if args.show:
            cv2.imshow("eye{!s}".format(num), i_img)            
        num += 1

    cv2.waitKey(0) 
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", "-c", default=None, help="Camera ID to use for pulling video from connected camera. Defaults to None.")
    parser.add_argument("--filename", "-f", default=None, help="Filename of image to process")
    parser.add_argument("--output", "-o", default="./", help="Directory to write output files to. Output files retain the original filename but append '_iris' to the end before the extension")
    parser.add_argument("--show", default=False, help="If true will show images of the identified eyes", action="store_true")

    args = parser.parse_args()
    main(args)
