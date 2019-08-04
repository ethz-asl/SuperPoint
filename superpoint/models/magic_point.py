import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import detector_head, detector_loss, box_nms
from .homographies import homography_adaptation_multispectral

import logging
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

class MagicPoint(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32},
            'image_ir': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_first',
            'kernel_reg': 0.,
            'grid_size': 8,
            'detection_threshold': 0.4,
            'homography_adaptation': {'num': 0},
            'nms': 0,
            'top_k': 0
    }

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)
        image_opt = inputs['image']
        image_ir = inputs['image_ir']

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            features = vgg_backbone(image, **config)
            outputs = detector_head(features, **config)
            return outputs

        def pass_opt_image():
          logging.info("Pass opt image called.")
          return image_opt
 
        def pass_ir_image():
          logging.info("Pass ir image called.")
          return image_ir

        p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        pred = tf.less(p_order, 0.5)
        image_to_use = tf.cond(pred, pass_opt_image, pass_ir_image)

        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            # Used only when exporting pseudo ground-truth
            logging.info("Generating net outputs by applying multispectral homography adaption")
            outputs = homography_adaptation_multispectral(image_opt, image_ir, net, config['homography_adaptation'])
        elif (mode == Mode.PRED) and config['homography_adaptation']['num'] == 0:
            # If no homographic adaption, use optical image (special case for exporting repeatability scores)
            logging.info("Generating net outputs by using optical image only (repeatability special case)")
            outputs = net(image_to_use)
        else:
            # Compute interest points in optical image only
            logging.info("Generating net outputs by using optical image only (during training)")
            outputs = net(image_to_use)

        prob = outputs['prob']
        if config['nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
                                               min_prob=config['detection_threshold'],
                                               keep_top_k=config['top_k']), prob)
            outputs['prob_nms'] = prob
        pred = tf.to_int32(tf.greater_equal(prob, config['detection_threshold']))
        outputs['pred'] = pred

        return outputs

    def _loss(self, outputs, inputs, **config):
        if config['data_format'] == 'channels_first':
            outputs['logits'] = tf.transpose(outputs['logits'], [0, 2, 3, 1])
        return detector_loss(inputs['keypoint_map'], outputs['logits'],
                             valid_mask=inputs['valid_mask'], **config)

    # TODO: add involved corner detector metrics
    def _metrics(self, outputs, inputs, **config):
        pred = outputs['pred']
        labels = inputs['keypoint_map']

        precision = tf.reduce_sum(pred*labels) / tf.reduce_sum(pred)
        recall = tf.reduce_sum(pred*labels) / tf.reduce_sum(labels)

        return {'precision': precision, 'recall': recall}
