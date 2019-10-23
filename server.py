# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/10

import re
import os
import json
import time
import torch
from flask import Flask, request, jsonify
from utils.images_utils import img_size_process
from utils.skew_correct import correct_image
from utils.server_utils import get_data
from utils.logserver import LogServer


"""
Parts of the codes form https://git.iyunxiao.com/DeepVision/text-detection-ctpn/blob/deploy_CT2D/main_server.py
by mcm
"""

app = Flask(__name__)

# 加载识别模型
# recg_model_name = "resnet"
# # recg_model_name = "crnn"
# if recg_model_name == "crnn":
#     model = load_reco_model(config.reco_model_path, config.class_num)
#     convert_list = load_convert_list(config.convert_list_path)
#     converter = utils_cn.strLabelConverter(config.convert_list_path, ignore_case=False)
# elif recg_model_name == "resnet":
#     model, converter = load_res_model(config.res_vocab_path, config.res_recg_model_path)
# ## 加载检测模型
# # Text_detector = TextLineDetector()
# text_detector = load_det_model(config.det_model_path)

# 创建过滤对象
# s = DelError(False)

# 日志文件
log_server = LogServer(app='ocr-printed', log_path=config.log_path)

max_w = 3000  # 输入检测网络图像的最大宽度
max_h = 2100  # 输入检测网络图像的最大高度
max_pixel_sum = max_w * max_h  # 输入检测网络图像的最大像素和

# 公式识别接口
formula_url = config.formula_url
print("ALL IS DONE")

#pipeline
from .pipeline.ocr import OCRPipeLine
#检测模型及识别模型都在PipeLine 里面加载了
ocr_pipeline = OCRPipeLine()





@app.route('/get_ocr_result_v2', methods=["POST"])
def main_v2():
    all_time_s1 = time.time()

    # 获取并解释post的数据
    code, message = get_data(request)
    if code:
        predictions = {"code": code, "message": message}
        return jsonify(predictions)
    im_url, im_str, subject, _type, from_src, color_img = message

    # 根据类型对图片大小进行处理
    code, message = img_size_process(color_img, _type)
    if code:
        predictions = {"code": code, "message": message}
        return jsonify(predictions)

    color_img, scale = message

    # 旋转角度检测与倾斜校正
    s1 = time.time()
    color_img, angle = correct_image(color_img)
    s2 = time.time()
    log_server.logging('>>>>>>>> Time of correct image: {:.2f}'.format(s2 - s1))

    ## AB卷OCR识别的预处理
    # if _type == "AB_OCR":
    #    color_img = dec_image_preprocess_ABOCR(color_img, side=1200)

    # 记录输入网络的图片大小(resize+倾斜校正)
    log_server.logging('>>>>>>>> Size of image: {}x{}'.format(color_img.shape[0], color_img.shape[1]))

    # 对数学学科进行公式识别
    if subject in ("数学", "物理", "化学", "生物"):
        s1 = time.time()
        formula_boxes, formula_labels, color_img = formula_det_reco(color_img, from_src, subject)
        s2 = time.time()
        log_server.logging('>>>>>>>> Time of formula: {:.2f}'.format(s2 - s1))
    else:
        formula_boxes, formula_labels = [], []

    # pixellink检测
    s1 = time.time()
    # boxes_coord = one_image_dec(color_img, Text_detector)
    boxes_coord = ocr_pipeline.det_model.infer(color_img)
    s2 = time.time()
    log_server.logging('>>>>>>>> Time of detection: {:.2f}'.format(s2 - s1))

    # 结合公式检测的结果对检测框进行切分
    boxes_coord = split_boxes(formula_boxes, boxes_coord)
    # 保证xmin<xmax,ymin<ymax
    boxes_coord = list(filter(lambda lst: lst[0] < lst[2] and lst[1] < lst[3], boxes_coord))

    ## 转换为crnn的输入格式
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    # gray_img = color_img

    with torch.no_grad():
        # crnn识别
        if recg_model_name == "crnn":
            labellist = call_crnn(gray_img, boxes_coord, subject)
        elif recg_model_name == "resnet":
            labellist = call_resnet(gray_img, boxes_coord, model, converter, subject)

    # 识别结果过滤
    no_del_subjects = ["英语"]
    if subject not in no_del_subjects:  # 化学和英语不处理
        s1 = time.time()
        labellist = list(map(lambda x: s.segment(x), labellist))
        s2 = time.time()
        log_server.logging('>>>>>>>> Time of filter: {:.2f}'.format(s2 - s1))

    # 进一步处理个别字符没有被检测为公式
    if subject in ("数学", "物理", "化学", "生物"):
        for i in range(len(labellist)):
            # print(labellist[i])
            labellist[i] = textExtrFormula(labellist[i].strip(), subject=subject)
            # print(labellist[i])
            # print("--"*20)

    # 英语分词处理
    # if subject == "英语" and False:
    #     s1 = time.time()
    #     labellist = list(map(lambda x: split_space.split(x), labellist))
    #     s2 = time.time()
    #     log_server.logging('>>>>>>>> Time of split word filter: {:.2f}'.format(s2 - s1))

    # 全半角符号转换
    # if subject!='英语':
    s1 = time.time()
    labellist = list(map(lambda x: symbol_transfer_ada.replace_symbol_ada(x, subject), labellist))
    s2 = time.time()
    log_server.logging('>>>>>>>> Time of symbol transfer ada filter: {:.2f}'.format(s2 - s1))

    # 末尾圆括号对缺失处理
    # if subject != '英语':
    s1 = time.time()
    labellist = list(map(lambda x: symbol_lack_detect.fill_tail_lackV2(x), labellist))
    s2 = time.time()
    log_server.logging('>>>>>>>> Time of symbol_lack_detect.fill_tail_lack filter: {:.2f}'.format(s2 - s1))

    # 还原到ctpn输入图片的坐标(注意是resize+倾斜校正后的图像坐标)
    if _type != "paper_segment":
        boxes_coord = np.asarray(boxes_coord) / scale
        boxes_coord = boxes_coord.astype(int)
        boxes_coord = boxes_coord.tolist()

    # 添加公式检测的结果
    boxes_coord.extend(formula_boxes)
    labellist.extend(formula_labels)
    boxes_coord = np.asarray(boxes_coord, dtype=int).tolist()

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