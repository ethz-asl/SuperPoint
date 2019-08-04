import numpy as np
import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from .utils import pipeline
from superpoint.settings import DATA_PATH, EXPER_PATH

import random
import logging


class CocoAndWarm(BaseDataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }

    def _init_dataset(self, **config):
        base_path_opt = Path(DATA_PATH, 'coco_and_warm/optical')
        image_paths_opt = sorted(list(base_path_opt.iterdir()))
        if config['truncate']:
            image_paths_opt = image_paths_opt[:config['truncate']]
        
        base_path_ir = Path(DATA_PATH, 'coco_and_warm/thermal')
        image_paths_ir = sorted(list(base_path_ir.iterdir()))
        if config['truncate']:
            image_paths_ir = image_paths_ir[:config['truncate']]

	# Jointly reshuffle list
        combined_paths = list(zip(image_paths_opt, image_paths_ir))
        random.shuffle(combined_paths)
        image_paths_opt[:], image_paths_ir[:] = zip(*combined_paths)

        names_opt = [p.stem for p in image_paths_opt]
        image_paths_opt = [str(p) for p in image_paths_opt]

        names_ir = [p.stem for p in image_paths_ir]
        image_paths_ir = [str(p) for p in image_paths_ir]
        
        files = {'image_paths': image_paths_opt, 'names': names_opt, 'image_paths_ir': image_paths_ir, 'names_ir': names_ir}

        if config['labels']:
            label_paths = []
            for n in names_opt:
                p = Path(EXPER_PATH + "/optical", config['labels'], '{}.npz'.format(n))
                assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
                label_paths.append(str(p))
            files['label_paths'] = label_paths

        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=config['num_parallel_calls'])

        return files

    def _get_data(self, files, split_name, **config):
        has_keypoints = 'label_paths' in files
        logging.info('Has keypoints bool = ' + str(has_keypoints))
        is_training = split_name == 'training'

        def _read_image(path):
            image = tf.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            return image

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_resize(image,
                                                         **config['preprocessing'])
            return image

        # Python function
        def _read_points(filename):
            return np.load(filename.decode('utf-8'))['points'].astype(np.float32)

        names_opt = tf.data.Dataset.from_tensor_slices(files['names'])
        images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
        images = images.map(_read_image)
        images_opt = images.map(_preprocess)
        names_ir = tf.data.Dataset.from_tensor_slices(files['names_ir'])
        images = tf.data.Dataset.from_tensor_slices(files['image_paths_ir'])
        images = images.map(_read_image)
        images_ir = images.map(_preprocess)
        data = tf.data.Dataset.zip({'image': images_opt, 'name': names_opt, 'image_ir': images_ir, 'name_ir': names_ir})

        # Add keypoints (should only have one set of keypoints, they are always multispectral)
        if has_keypoints:
            kp = tf.data.Dataset.from_tensor_slices(files['label_paths'])
            kp = kp.map(lambda path: tf.py_func(_read_points, [path], tf.float32))
            kp = kp.map(lambda points: tf.reshape(points, [-1, 2]))
            data = tf.data.Dataset.zip((data, kp)).map(
                    lambda d, k: {**d, 'keypoints': k})
            data = data.map(pipeline.add_dummy_valid_mask)

        # Keep only the first elements for validation
        if split_name == 'validation':
            data = data.take(config['validation_size'])

        # Cache to avoid always reading from disk
        if config['cache_in_memory']:
            tf.logging.info('Caching data, fist access will take some time.')
            data = data.cache()

        # Generate the warped pair (relevant for SuperPoint, not MagicPoint)
        if config['warped_pair']['enable']:
            assert has_keypoints
            warped = data.map_parallel(lambda d: pipeline.homographic_augmentation_multispectral(
                d, add_homography=True, **config['warped_pair']))
            if is_training and config['augmentation']['photometric']['enable']:
                warped = warped.map_parallel(lambda d: pipeline.photometric_augmentation_multispectral(
                    d, **config['augmentation']['photometric']))
            warped = warped.map_parallel(pipeline.add_keypoint_map)
            # Merge with the original data
            data = tf.data.Dataset.zip((data, warped))
            data = data.map(lambda d, w: {**d, 'warped': w})

        # Data augmentation (relevant for SuperPoint and MagicPoint training, not experorting detections)
        logging.info('Is training bool = ' + str(is_training))
        if has_keypoints and is_training:
            if config['augmentation']['photometric']['enable']:
                data = data.map_parallel(lambda d: pipeline.photometric_augmentation_multispectral(
                    d, **config['augmentation']['photometric']))
            if config['augmentation']['homographic']['enable']:
                assert not config['warped_pair']['enable']  # doesn't support hom. aug.
                data = data.map_parallel(lambda d: pipeline.homographic_augmentation_multispectral(
                    d, **config['augmentation']['homographic']))

        # Generate the keypoint map
        if has_keypoints:
            data = data.map_parallel(pipeline.add_keypoint_map)

        return data
