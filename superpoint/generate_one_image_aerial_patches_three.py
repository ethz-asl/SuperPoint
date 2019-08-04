import numpy as np
import tensorflow as tf
import cv2 as cv
import os
import argparse
import yaml
from pathlib import Path

from superpoint.models.homographies import (sample_homography, flat2mat,
                                            invert_homography)
from superpoint.settings import DATA_PATH


seed = None


def _scale_preserving_resize(image):
    target_size = tf.convert_to_tensor(config['preprocessing']['resize'])
    scales = tf.to_float(tf.divide(target_size, tf.shape(image)[:2]))
    new_size = tf.to_float(tf.shape(image)[:2]) * tf.reduce_max(scales)
    image = tf.image.resize_images(image, tf.to_int32(new_size),
                                   method=tf.image.ResizeMethod.BILINEAR)
    return tf.image.resize_image_with_crop_or_pad(image, target_size[0],
                                                  target_size[1])


def _preprocess(image):
    image = tf.image.rgb_to_grayscale(image)
    if config['preprocessing']['resize']:
        image = _scale_preserving_resize(image)
    return image


if __name__ == '__main__':
    tf.set_random_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default=None)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    base_path = Path(DATA_PATH, 'one_image_aerial_three/optical')
    base_path_ir = Path(DATA_PATH, 'one_image_aerial_three/thermal')
    image_paths = sorted(list(base_path.iterdir()))
    image_paths_ir = sorted(list(base_path_ir.iterdir()))
    output_dir = Path(DATA_PATH, 'one_image_aerial_three/patches/')
    if not output_dir.exists():
        os.makedirs(output_dir)

    # Create the ops to warp an image
    tf_path = tf.placeholder(tf.string)
    # Read the image
    image = tf.read_file(tf_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = _preprocess(image)
    shape = tf.shape(image)[:2]
    tf_path_ir = tf.placeholder(tf.string)
    image_ir = tf.read_file(tf_path_ir)
    image_ir = tf.image.decode_jpeg(image_ir, channels=3)
    image_ir = _preprocess(image_ir)

    # Warp the image
    H = sample_homography(tf.shape(image)[:2], **config['homographies'])
    warped_image = tf.contrib.image.transform(image_ir, H, interpolation="BILINEAR")
    patch_ratio = config['homographies']['patch_ratio']
    new_shape = tf.multiply(tf.cast(shape, tf.float32), patch_ratio)
    new_shape = tf.cast(new_shape, tf.int32)
    warped_image = tf.image.resize_images(warped_image, new_shape)
    H = invert_homography(H)
    H = flat2mat(H)[0, :, :]

    print("Generating patches of Warm val...")
    sess = tf.InteractiveSession()
    for num, path in enumerate(image_paths):
        path_ir = image_paths_ir[num]
        new_path = Path(output_dir, "v_" + str(num))
        if not new_path.exists():
            os.makedirs(new_path)

        # Run
        im, im_ir, warped_im, homography = sess.run([image, image_ir, warped_image, H],
                                             feed_dict={tf_path: str(path), tf_path_ir: str(path_ir)})

        # Add scaling to adapt to the fact that the patch is
        # twice as small as the original image
        homography[2, :] /= patch_ratio

        # Write the result in files
        cv.imwrite(str(Path(new_path, "1.jpg")), im)
        cv.imwrite(str(Path(new_path, "1_ir.jpg")), im_ir)
        cv.imwrite(str(Path(new_path, "2.jpg")), warped_im)
        np.savetxt(Path(new_path, "H_1_2"), homography, '%.5g')
    print("Files generated in " + str(output_dir))
