import sys
sys.path.append("..")
from configs.config_util import ConfigDict
from configs.resbilstm import config as resbilstm_config
from tools import InferModel
import os
import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2

def visual(imgpath, true_label, pred_label, image_font, desdir):
    img = Image.open(imgpath).convert('RGB')
    w, h = img.size
    bg = Image.new('RGB', (2 * w, int(4.5 * h)), (255, 255, 255))
    img_draw = ImageDraw.Draw(bg)
    bg.paste(img, (0, int(3.5*h)))
    img_draw.text((5, 5), pred_label, fill=(0, 0, 255), font=image_font)
    img_draw.text((5, int(1.5*h)), true_label, fill=(0, 255, 0), font=image_font)
    basename = os.path.basename(imgpath)
    bg.save(os.path.join(desdir, basename))





resbilstm_args = ConfigDict(resbilstm_config)
resbilstm_args.isTrain = False

reg_model = InferModel(resbilstm_args)  # 识别模型


def test_accuracy():
    datadir = r"/media/Data/hcn/data/student_images_all_det/row_data_1_wrong/test_data"
    desdir = r"/media/Data/hcn/data/student_images_all_det/row_data_1_wrong/test_data_wrong"
    labelfile = r"/media/Data/hcn/data/student_images_all_det/row_data_1_wrong/test_data/label.txt"
    wrong_type = {'.': '1', ',': '2', ']': '3'}
    for item in wrong_type:
        if not os.path.exists(os.path.join(desdir, wrong_type[item])):
            os.mkdir(os.path.join(desdir, wrong_type[item]))
    if not os.path.exists(os.path.join(desdir, '0')):
        os.mkdir(os.path.join(desdir, '0'))
    lines = open(labelfile, 'r', encoding='utf-8').read().strip('\n').split('\n')
    allnum = len(lines)
    right_count = 0
    tmp_count = 0
    image_font = ImageFont.FreeTypeFont(font='STKAITI.TTF', size=32)
    for line in tqdm.tqdm(lines):
        savedir = os.path.join(desdir, '0')
        try:
            tmp_count += 1
            imgpath, label = line.split()
            img = cv2.imread(os.path.join(datadir, imgpath))
            pred_label = reg_model.infer(img, subject="历史")
            # pred_label = pred_label.replace(':', '：')
            # pred_label = pred_label.replace('(', '（')
            # pred_label = pred_label.replace(')', '）')
            if label.strip() == pred_label.strip():
                right_count += 1
            else:
                for item in wrong_type:
                    f_label = pred_label.replace(item, '')
                    if label.strip() == f_label.strip():
                        savedir = os.path.join(desdir, wrong_type[item])
                        break
                visual(os.path.join(datadir, imgpath), label.strip(), pred_label, image_font, savedir)
            if not tmp_count == 0 and tmp_count % 100 == 0:
                print("right/num:{}/{} accuracy:{}".format(right_count, tmp_count, right_count / tmp_count))
        except:
            print(imgpath + ": have problem!")

def get_wrong_type():
    datadir = r"/home/moveDir/hcn/paper_structure/tihao/digits"
    wrongdir = r"/home/moveDir/hcn/paper_structure/tihao/wrong"
    desdir = r"/home/moveDir/hcn/paper_structure/tihao/wrong_type"
    imgnames = os.listdir(wrongdir)
    image_font = ImageFont.FreeTypeFont(font='STKAITI.TTF', size=32)
    wrong_type = {'.': '1', ',': '2', ']': '3'}
    for item in wrong_type:
        if not os.path.exists(os.path.join(desdir, wrong_type[item])):
            os.mkdir(os.path.join(desdir, wrong_type[item]))
    if not os.path.exists(os.path.join(desdir, '0')):
        os.mkdir(os.path.join(desdir, '0'))
    for imgname in tqdm.tqdm(imgnames):
        imgpath = os.path.join(datadir, imgname)
        label = imgname.split('.')[0].split('-')[-1]
        img = cv2.imread(os.path.join(datadir, imgpath))
        pred_label = reg_model.infer(img, subject="历史")
        savedir = os.path.join(desdir, '0')
        for item in wrong_type:
            f_label = pred_label.replace(item, '')
            if label.strip() == f_label.strip():
                savedir = os.path.join(desdir, wrong_type[item])
                break
        visual(imgpath, pred_label, image_font, savedir)


if __name__ == '__main__':
    test_accuracy()




