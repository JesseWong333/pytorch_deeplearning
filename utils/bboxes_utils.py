"""
检测方框的后处理相关函数，如排序，框分割
"""
import numpy as np
import operator
def process_bboxes(bboxes, h, w):
    """
    对检测框再做一些后处理。包括 边框超出边界的处理，边缘的滤除，边框的缩放，
    :param bboxes:
    :return:
    """
    # 要考虑是否为0的情形
    bboxes = [(max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])) for bbox in bboxes]
    bboxes = set(bboxes)
    # 上边界的比例， 下边界的比例. 宽度为32像素的图，实际文字大小只有24. 上下只扩充实际大小的1/6
    bboxes_p = set()
    for bbox in bboxes:
        #expand = (bbox[3] - bbox[1])/6
        expand = 4
        #bboxes_p.add((bbox[0], max(0, bbox[1]-expand),  bbox[2], min(h, bbox[3]+expand)))
        #bboxes_p.add((max(0, bbox[0]-2), max(0, bbox[1]-expand),  min(w, bbox[2]+4), min(h, bbox[3]+expand)))
        bboxes_p.add((bbox[0], max(0, bbox[1] - expand), bbox[2], min(h, bbox[3] + expand)))
    bboxes_p = np.array(list(bboxes_p), dtype=int)
    return bboxes_p

def sort_boxes(iboxes_coord):
    """
    对检测框进行排列
    :param iboxes_coord: input boxes coordinate
    :return:
    oboxes_coord, output sorted boxes coordinate
    """
    oboxes_coord = []
    if not iboxes_coord:  # 保证检测框个数>=1
        return oboxes_coord

    # 按检测框的中心y点排序
    y_centers = {}
    for count, coord in enumerate(iboxes_coord):
        y_center = (coord[1]+coord[3]) // 2     # ymin,ymax = coord[1],coord[3]
        y_centers[count] = y_center
    y_centers = sorted(y_centers.items(), key=operator.itemgetter(1))

    lines = []  # 总共的行集合列表
    line = []  # 高度接近的检测框行集合列表
    line.append(y_centers[0])  # 初始化第一行
    for y_center in y_centers[1:]:
        idx, y_c = y_center  # idx表示第几个检测框,y_c表示该检测框的y轴中心点
        h1 = (iboxes_coord[line[-1][0]][3]- iboxes_coord[line[-1][0]][1])/2
        h2 = (iboxes_coord[idx][3] - iboxes_coord[idx][1]) / 2
        h = h1 if h1 > h2 else h2
        # 如果当前中心y点与上一个检测框的中心y点的差小于当前框的高度的一半
        if (y_c - line[-1][1]) < (h):
        #if (y_c - line[-1][1]) < ((iboxes_coord[idx][3] - iboxes_coord[idx][1]) / 2):
            line.append(y_center)
        else:
            lines.append(line)
            line = []
            line.append(y_center)
    lines.append(line) #分行结果

    for count, line in enumerate(lines):
        if len(line) == 1:
            oboxes_coord.append(iboxes_coord[line[0][0]])
        else:
            orilineboxes = {}  # 同一个行内多个检测框的排序
            for idx, line_item in enumerate(line):  # line_item: 一行的每个检测框xmin
                orilineboxes[idx] = iboxes_coord[line_item[0]]
            orilineboxes = sorted(orilineboxes.items(), key=operator.itemgetter(1))
            for orilinebox in orilineboxes:
                idx, coord = orilinebox
                oboxes_coord.append(coord)

    return oboxes_coord


def split_boxes(formula_boxes, commom_boxes):
    """
    根据公式检测框对文字检测框进行分框
    避免后面的检测框side-refinement不准确,记录检测框的中点
    """
    if not len(formula_boxes):
        return commom_boxes

    get_midx = lambda box: (box[0]+box[2])//2
    get_midy = lambda box: (box[1]+box[3])//2
    get_boxw = lambda box: box[2]-box[0]
    get_boxh = lambda box: box[3]-box[1]
    f = lambda box: (get_midx(box), get_midy(box), get_boxw(box), get_boxh(box))

    new_commom_boxes = []
    for cbox in commom_boxes:
        split_points = []       # 记录分界点位置
        cbox_midx, cbox_midy, cbox_w, cbox_h = f(cbox)
        cbox_minx, cbox_miny, cbox_maxx, cbox_maxy = cbox

        for fbox in formula_boxes:
            fbox_midx, fbox_midy, fbox_w, fbox_h = f(fbox)

            # 使用排序的策略，先确定两个box是否可能属于一行
            max_h2 = max(cbox_h, fbox_h)//2
            if abs(cbox_midy-fbox_midy) < max_h2:   # 满足条件的认为属于一行
                #http://yx-yuejuan.ks3-cn-beijing.ksyun.com/template/0/2E242492-9104-11E9-91CE-4CCC6A459612.png@base@tag=imgScale&c=1&f=1&cox=222&coy=1318&w=1223&h=144?KSSAccessKeyId=AKLTu3UeHldJSaGxjdNRU3MomQ&Expires=10200827670&Signature=NW6marW3IWzXHdnASIOgndzmZoM=
                if (fbox_midx-cbox_minx) * (fbox_midx-cbox_maxx) < 0:   # fbox_midx在(cbox_minx,cbox_maxx之间)  # 有问题
                    split_points.append(fbox_midx)

        if not split_points:
            new_commom_boxes.append(cbox)
        else:
            split_points.sort()
            split_points.insert(0, cbox_minx)
            split_points.append(cbox_maxx)
            new_commom_boxes.extend([[split_points[i], cbox_miny, split_points[i+1], cbox_maxy] for i in range(len(split_points)-1)])

    return new_commom_boxes



