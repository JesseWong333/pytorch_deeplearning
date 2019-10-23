
import sys
sys.path.append("..")

from configs.config_util import ConfigDict
from tools import InferModel
import cv2
import torch

def maybe_resize(img):
    # src_img = input.copy()
    ratio = 1
    if max(img.shape[:2]) > 3200:
        if img.shape[0] > img.shape[1]:
            ratio = 3200 / img.shape[0]
        else:
            ratio = 3200 / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
    return img, ratio

class FormulaPipeLine(object):
    def __init__(self, ):
        from configs.im2latex import config as im2text_config
        from configs.pixel_link import config as pixellink_config

        pixellink_args = ConfigDict(pixellink_config)
        pixellink_args.isTrain = False
        self.det_model = InferModel(pixellink_args)


        im2latex_args = ConfigDict(im2text_config)
        im2latex_args.isTrain = False
        self.reg_model = InferModel(im2latex_args)

    def infer(self, img): #单个例子
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w, _ = img.shape
        img, ratio = maybe_resize(img)
        cords = self.det_model.infer(img, ratio=ratio, src_img_shape=img.shape)
        final_cords = []
        final_formula = []
        for cord in cords:
            imgcrop = img_gray[cord[1]:cord[3], cord[0]:cord[2]]
            formula = self.reg_model.infer(imgcrop)
            if len(formula) > 0:
                final_cords.append(cord)
                final_formula.append(formula)
        return final_cords, final_formula



if __name__ == '__main__':
    import torchvision.transforms.functional as F
    formula_pipeline = FormulaPipeLine()
    img_path = '32067_20.png'
    img = cv2.imread(img_path)
    img_tensor1 = torch.from_numpy(img)
    img_tensor2 = F.to_tensor(img)
    img_tensor3 = torch.FloatTensor(img)
    print(img)
    print(img_tensor1)
    print(img_tensor2)
    print(img_tensor3)
    # cords, formulas = formula_pipeline.infer(img)
    # print(cords)
    # print(formulas)











