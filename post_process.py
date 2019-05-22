import numpy as np
from scipy.special import softmax
from losses.prior_box_eval import PriorBoxEval
from detection_layer import DetectHead, DetectTail
from utils.nms_wrapper import nms


cfg = {
    'feature_maps' : [(32, 64)],
    'min_dim' : (512, 1024),
    'steps' : [16],
    # 在15~70之间均匀
    'defaultbox': [
        [16, 16],
        [27, 27],
        [30, 30],
        [40, 40],
        [50, 50],
        [60, 60],
        [70, 70]],
    'variance' : [0.1, 0.2],
    'clip' : False,
}


# 分别从头和尾找， 当两者都认为自己找到了最好的，才是一个pair. This is love
# we call this algorithm as "loving match"

def find_from_head(c_dets_head, c_dets_tail):
    no_matched_head = []
    matched_pairs = set()
    for head_index in range(c_dets_head.shape[0]):
        xmin_H, ymin_H, xmax_H, ymax_H, _, _ = c_dets_head[head_index]
        macthed_tails = []

        # 怎样保证在复杂的真实环境中可以匹配得更好? 看弧度，
        for tail_index in range(c_dets_tail.shape[0]):
            xmin_T, ymin_T, xmax_T, y_max_T, _, _ = c_dets_tail[tail_index]

            h = max(tail_h, head_h)

            yy1 = np.minimum(left_down[1], right_down[1])  # 应该是要看高的相差程度s
            yy2 = np.maximum(left_up[1], right_up[1])
            overlap_h = np.maximum(0, yy1-yy2 + 1) / h

            # tail应当是 右边 且 距离最近的一个.  必须要两者都满足， 才确定是合适的tail, 否则再通过文本内部点找
            if overlap_h > 0.2 and tail_w_ave - head_w_ave > 0:
                macthed_tails.append((tail_index, overlap_h, tail_w_ave - head_w_ave))

        if len(macthed_tails) == 0:
            no_matched_head.append(head_index)
            continue

        # todo: 这样写会造成问题，有点先来后到的意思， 第一个满足了，但是第二个更满足.

        # best_macthed_tail = macthed_tails[0]
        # for macthed_tail in macthed_tails[1:]:  # 觉得最"近"重要一些， overlap_h 只要相差不大就可以了
        #     if macthed_tail[2] <= best_macthed_tail[2]:
        #         if macthed_tail[1] - best_macthed_tail[1] > -0.1:
        #             best_macthed_tail = macthed_tail

        # 代码堆得太多了，再将候选的匹配按照“弧度”的大小进行匹配
        for macthed_tail in macthed_tails:
            pass

        # matched_pairs.add((head_index, best_macthed_tail[0]))
    return matched_pairs, no_matched_head


# todo: 这个函数其实跟从头找是一样的， 可以考虑合并. 目前这样主要是为了理清楚逻辑
def find_from_tail(head_cord_list, tail_cord_list):
    no_matched_tail = []
    matched_pairs = set()
    for tail_index in range(tail_cord_list.shape[0]):
        right_up = tail_cord_list[tail_index][1]
        right_down = tail_cord_list[tail_index][0]
        tail_h = right_down[1] - right_up[1]
        tail_w_ave = (right_down[0] + right_up[0]) / 2
        macthed_heads = []
        for head_index in range(head_cord_list.shape[0]):

            left_up = head_cord_list[head_index][0]
            left_down = head_cord_list[head_index][1]
            head_h = left_down[1] - left_up[1]
            head_w_ave = (left_up[0] + left_down[0]) / 2

            h = min(tail_h, head_h)

            yy1 = np.minimum(left_down[1], right_down[1])
            yy2 = np.maximum(left_up[1], right_up[1])
            overlap_h = np.maximum(0, yy1-yy2 + 1) / h

            if overlap_h > 0.5 and tail_w_ave - head_w_ave > 0:
                macthed_heads.append((head_index, overlap_h, tail_w_ave - head_w_ave))

        if len(macthed_heads) == 0:
            no_matched_tail.append(tail_index)
            continue

        best_macthed_head = macthed_heads[0]
        for macthed_head in macthed_heads[1:]:  # 觉得最"近"重要一些， overlap_h 只要相差不大就可以了
            if macthed_head[2] <= best_macthed_head[2]:
                if macthed_head[1] - best_macthed_head[1] > -0.1:
                    best_macthed_head = macthed_head
        matched_pairs.add((best_macthed_head[0], tail_index))
    return matched_pairs, no_matched_tail


def thresh_bboxes(boxes, scores, width, heigth, num_classes=2, thresh=0.05):
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    scores = softmax(scores, axis=2)
    # scale each detection back up to the image
    scale = [width, heigth, width, heigth]
    boxes *= scale

    # 处理box
    batch_dets = []
    for batch in range(boxes.shape[0]):
        for j in range(1, num_classes):
            inds = np.where(scores[batch, :, j] > thresh)[0]   #  所有的score没有经过softmax之类的东西， why?
            c_bboxes = boxes[batch, inds]
            c_scores = scores[batch, inds, j]
            label = np.ones((inds.shape[0], 1)) * j
            c_dets_class = np.hstack((c_bboxes, c_scores[:, np.newaxis], label)).astype(
                np.float32, copy=False)
            if j == 1:
                c_dets = c_dets_class
            else:
                c_dets = np.concatenate((c_dets, c_dets_class))
        batch_dets.append(c_dets)
    return batch_dets


def post_process(pre, h, w):
    # 创建prior
    cfg['feature_maps'] = [(h//16, w//16)]
    cfg['min_dim'] = (h, w)
    priorbox = PriorBoxEval(cfg)
    priors = priorbox.forward()

    detector_head = DetectHead(2, 0, cfg)
    boxes_head, scores_head = detector_head.forward(pre, priors, h, w)  # score部分从头到尾都没有经过nms

    detector_tail = DetectTail(2, 0, cfg)
    boxes_tail, scores_tail = detector_tail.forward(pre, priors, h, w)

    batch_dets_head = thresh_bboxes(boxes_head, scores_head, w, h)
    batch_dets_tail = thresh_bboxes(boxes_tail, scores_tail, w, h)

    # 进行nms
    batch_dets_head = batch_dets_head[0]
    keep = nms(batch_dets_head, 0.1, force_cpu=True)  # 存储了保存的
    c_dets_head = batch_dets_head[keep, :]

    batch_dets_tail = batch_dets_tail[0]
    keep = nms(batch_dets_tail, 0.1, force_cpu=True)
    c_dets_tail = batch_dets_tail[keep, :]

    # 从头去找， 尾可能会匹配两次. 从尾去找，头可能会匹配两次。
    # matched_pairs_from_head, no_matched_head = find_from_head(c_dets_head, c_dets_tail)
    # matched_pairs_from_tail, no_matched_tail = find_from_tail(c_dets_head, c_dets_tail)
    # matched_pairs = matched_pairs_from_head & matched_pairs_from_tail

    return c_dets_head, c_dets_tail