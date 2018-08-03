import logging

import slidingwindow as sw

import cv2
import numpy as np
import tensorflow as tf
import time

from tf_pose import common
from tf_pose.common import CocoPart
from tf_pose.tensblur.smoother import Smoother
import tensorflow.contrib.tensorrt as trt
try:
    from tf_pose.pafprocess import pafprocess
except ModuleNotFoundError as e:
    print(e)
    print('you need to build c++ library for pafprocess. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess')
    exit(-1)

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def parts(self):
	return self.body_parts

    def part_nums(self):
	return self.body_parts.keys()

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()

class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def getData(self):
	return (self.x, self.y, self.score)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()

    '''def partData(self):
	return [self.part_idx, self.x, self.y, self.score]'''


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat):
        pafprocess.process_paf(peaks, heat_mat, paf_mat)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                    pafprocess.get_part_score(c_idx)
                )

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans


class TfPoseEstimator:
    # TODO : multi-scale

    def __init__(self, graph_path, target_size=(320, 240), tf_config=None):
        self.target_size = target_size

        # load graph
        logger.info('loading graph from %s(default size=%dx%d)' % (graph_path, target_size[0], target_size[1]))
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        trt_graph = trt.create_inference_graph(input_graph_def=graph_def,outputs=["Openpose/concat_stage7"], max_batch_size=1,max_workspace_size_bytes=1 << 25,precision_mode='FP16',minimum_segment_size=50)
	print(trt_graph)
        tf.import_graph_def(trt_graph, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph, config=tf_config)
        writer = tf.summary.FileWriter('/tmp/estimator',self.persistent_sess.graph)
        writer.flush()
        # for op in self.graph.get_operations():
        #     print(op.name)
        # for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
        #     print(ts)
        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.tensor_heatMat = self.tensor_output[:, :, :, :19]
        self.tensor_pafMat = self.tensor_output[:, :, :, 19:]
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')
        self.tensor_heatMat_up = tf.image.resize_area(self.tensor_output[:, :, :, :19], self.upsample_size,
                                                      align_corners=False, name='upsample_heatmat')
        self.tensor_pafMat_up = tf.image.resize_area(self.tensor_output[:, :, :, 19:], self.upsample_size,
                                                     align_corners=False, name='upsample_pafmat')
        smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()

        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                     tf.zeros_like(gaussian_heatMat))

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(tf.variables_initializer(
            [v for v in tf.global_variables() if
             v.name.split(':')[0] in [x.decode('utf-8') for x in
                                      self.persistent_sess.run(tf.report_uninitialized_variables())]
             ])
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1], target_size[0]]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 2, target_size[0] // 2]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 4, target_size[0] // 4]
            }
        )

    def __del__(self):
        # self.persistent_sess.close()
        pass

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2 ** 8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
    def draw_humans(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        for human in humans:
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
                cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

        return npimg

    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(h), self.target_size[1] / float(w)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1], 0.2)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            window_step = scale[1]

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1],
                                  window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped

    def inference(self, npimg, resize_to_default=True, upsample_size=1.0):
        if npimg is None:
            raise Exception('The image is not valid. Please check your image exists.')

        if resize_to_default: #1.08155608177

            upsample_size = [int(self.target_size[1] / 8 * upsample_size), int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size), int(npimg.shape[1] / 8 * upsample_size)]

        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)
            pass

        logger.debug('inference+ original shape=%dx%d' % (npimg.shape[1], npimg.shape[0]))
        img = npimg
        if resize_to_default:
            img = self._get_scaled_img(npimg, None)[0][0]
        peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], feed_dict={
                self.tensor_image: [img], self.upsample_size: upsample_size
            })
        peaks = peaks[0]
        self.heatMat = heatMat_up[0]
        self.pafMat = pafMat_up[0]
        logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (
        self.heatMat.shape[1], self.heatMat.shape[0], self.pafMat.shape[1], self.pafMat.shape[0]))

        t = time.time()
        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat)
        logger.debug('estimate time=%.5f' % (time.time() - t))
        return humans


if __name__ == '__main__':
    import pickle

    f = open('./etcs/heatpaf1.pkl', 'rb')
    data = pickle.load(f)
    logger.info('size={}'.format(data['heatMat'].shape))
    f.close()

    t = time.time()
    humans = PoseEstimator.estimate_paf(data['peaks'], data['heatMat'], data['pafMat'])
    dt = time.time() - t;
    t = time.time()
    logger.info('elapsed #humans=%d time=%.8f' % (len(humans), dt))
