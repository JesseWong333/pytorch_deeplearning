# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/10
import sys
sys.path.append("..")
import json
import time
import torch
from flask import Flask, request, jsonify
from utils.bboxes_utils import split_boxes, sort_boxes
from utils.images_utils import img_size_process, formula_det_reco
from utils.skew_correct import correct_image
from utils.text_utils import textExtrFormula
from server.server_utils import get_data, get_gpu_use
from server.logserver import LogServer
from pipeline.ocr import OCRPipeLine
import config
import cv2
import numpy as np

"""
Parts of the codes form https://git.iyunxiao.com/DeepVision/text-detection-ctpn/blob/deploy_CT2D/main_server.py
by mcm
"""
"""
服务的一些参数直接写在这里可以吗？还是新建一个config文件
"""

app = Flask(__name__)

# 加载识别模型

# 日志文件
log_server = LogServer(app='ocr-printed', log_path=config.log_path)

# 公式识别接口

#OCRPipeline
#检测模型及识别模型都在PipeLine 里面加载了
ocr_pipeline = OCRPipeLine()
print("ALL IS DONE")

@app.route('/')
def test():
    print("start success")

@app.route('/get_ocr_result_v2', methods=["POST"])
def main_v2():
    all_time_s1 = time.time()

    # 获取并解释post的数据
    code, message = get_data(request, log_server)
    if code:
        predictions = {"code": code, "message": message}
        return jsonify(predictions)
    im_url, im_str, subject, _type, from_src, color_img = message

    #将彩色图片转换为灰度图片，再转换为三通道图
    color_img_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    color_img = cv2.cvtColor(color_img_gray, cv2.COLOR_GRAY2BGR)
    #将彩色图片转换为灰度图片，再转换为三通道图

    # 根据类型对图片大小进行处理
    code, message = img_size_process(color_img, _type, log_server)
    if code:
        predictions = {"code": code, "message": message}
        return jsonify(predictions)

    color_img, scale = message

    # 旋转角度检测与倾斜校正
    s1 = time.time()
    color_img, angle = correct_image(color_img)
    s2 = time.time()
    log_server.logging('>>>>>>>> Time of correct image: {:.2f}'.format(s2 - s1))

    # 记录输入网络的图片大小(resize+倾斜校正)
    log_server.logging('>>>>>>>> Size of image: {}x{}'.format(color_img.shape[0], color_img.shape[1]))

    # 对理科学科进行公式识别
    if subject in ("数学", "物理", "化学", "生物"):
        s1 = time.time()
        formula_boxes, formula_labels, color_img, formula_message = formula_det_reco(color_img, log_server, from_src, subject)
        s2 = time.time()
        if formula_boxes==[]:#当公式识别结果返回为空时，打印公式识别接口返回状态消息，以便定位问题
            log_server.logging('>>>>>>>> status of formula: code:{} message:{}'.format(formula_message[0], formula_message[1]))
        log_server.logging('>>>>>>>> Time of formula: {:.2f}'.format(s2 - s1))
    else:
        formula_boxes, formula_labels = [], []

    #文字检测
    s1 = time.time()
    boxes_coord = ocr_pipeline.det_model.infer(color_img, ratio=1, src_img_shape=color_img.shape)
    s2 = time.time()
    log_server.logging('>>>>>>>> Time of text detection: {:.2f}'.format(s2 - s1))

    # 结合公式检测的结果对检测框进行切分
    boxes_coord = split_boxes(formula_boxes, boxes_coord)
    # 保证xmin<xmax,ymin<ymax
    boxes_coord = list(filter(lambda lst: lst[0] < lst[2] and lst[1] < lst[3], boxes_coord))

    ## 转换为识别模型的输入格式
    # gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    #文字识别
    s1 = time.time()
    labellist = ocr_pipeline.one_image_reg(color_img, boxes_coord, subject)
    s2 = time.time()
    log_server.logging('>>>>>>>> Time of text recognition: {:.2f}'.format(s2 - s1))

    # 进一步处理个别字符没有被检测为公式
    if subject in ("数学", "物理", "化学", "生物"):
        for i in range(len(labellist)):
            # print(labellist[i])
            labellist[i] = textExtrFormula(labellist[i].strip(), subject=subject)
            # print(labellist[i])
            # print("--"*20)

    # 添加公式检测的结果
    boxes_coord.extend(formula_boxes)
    labellist.extend(formula_labels)
    # boxes_coord = np.asarray(boxes_coord, dtype=int).tolist()

    # 还原到输入图片的坐标(注意是resize+倾斜校正后的图像坐标)
    if _type != "paper_segment":
        boxes_coord = np.asarray(boxes_coord) / scale
        boxes_coord = boxes_coord.astype(int)
        boxes_coord = boxes_coord.tolist()

    # 检测结果排序第二版本
    boxes_coord_t = sort_boxes(boxes_coord)
    sort_ids = [boxes_coord.index(box_t) for box_t in boxes_coord_t]
    labellist_t = np.array(labellist)[sort_ids]
    labellist_t = labellist_t.tolist()

    if get_gpu_use() > 5000:
        log_server.logging('>>>>>>>> gpu memory use more than 5000MB, clean now.')
        torch.cuda.empty_cache()

    predictions = {
        "message": "sucess",
        "code": 0,  # 请求返回状态
        "data": json.dumps({"boxes_coord": boxes_coord_t, "labellist": labellist_t})
    }
    all_time_s2 = time.time()
    log_server.logging('>>>>>>>> Time of total: {:.2f}'.format(all_time_s2 - all_time_s1))
    log_server.logging('*************END**************')

    return jsonify(predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7000)
    # app.run(host='0.0.0.0', port=9000, debug=False, use_reloader=False)