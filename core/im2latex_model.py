# coding: utf-8 --**--
"""
The codes are from https://git.iyunxiao.com/DeepVision/pytorch-ocr-framework

What You Get Is What You See: A Visual Markup Decompiler
im2latex
"""

import torch
import torch.nn as nn
import itertools
from core.losses.im2latex_loss import build_loss_compute
from core.modules.text import Vocab
from core.modules.beam_search import BeamSearch, GNMTGlobalScorer
from core.modules.im2latex_moudle import pad_batch_formulas
from core.modules.seq_seq_utils import tile
from .backbones.Im2Text import Im2TextModel
from .base_model import BaseModel
from . import register_model


import numpy as np
import os


@register_model
class Im2LatexModel(BaseModel):
    def name(self):
        return "Im2LatexModel"

    def initialize(self, args):
        BaseModel.initialize(self, args)

        self.loss_names = ['xent_loss']
        self.model_names = ['net',]
        self.output_names = ['results', ]
        model_cfg = args.Model
        vocab_cfg = args.Vocab
        self.vocab = Vocab(vocab_cfg)
        self.emb_dim = model_cfg.emb_dim
        self.enc_rnn_h = model_cfg.enc_rnn_h
        self.dec_rnn_h = model_cfg.dec_rnn_h
        self.vocab_size = self.vocab.n_tok
        self.id_end = self.vocab.id_end
        self._form_prepro = self.vocab.form_prepro
        self.max_length = 200
        self.n_best = 1
        self.replace_unk = False
        self.phrase_table = ""
        self.vis_flag = False
        self._exclusion_idxs = set()
        self.beam_size = 5
        self.global_scorer = GNMTGlobalScorer.from_opt(alpha=0.0, beta=-0.0, length_penalty='none',
                                                       coverage_penalty='none')

        self.net = Im2TextModel(self.vocab_size, self.vocab)

        self.net.to(self.gpu_ids[0])
        self.distributed = False #分布式训练参数，这个应该在配置文件里面有，但是现在框架还没考虑这个，所以
        if args.isTrain:
            self.optimizer = torch.optim.SGD(itertools.chain(self.net.parameters()),
                                             lr=args.lr)

            # 初始化损失函数, 检测的网络通常有很复杂的loss
            self.criterion = build_loss_compute(self.net)
            # 会调用父类的setup方法为optimizers设置lr schedulers
            self.optimizers = []
            self.optimizers.append(self.optimizer)

    def set_input(self, input,):
        if self.args.isTrain:
            if not self.distributed:
                self.batch_images = torch.FloatTensor(input[0]).to(self.device)
                formulas_list = []
                for i in input[1]:
                    tmp = list(self._form_prepro(i))
                    # 公式str预处理转int的label时, 在label 前插入init 的label
                    tmp.insert(0, self.vocab.id_init)
                    formulas_list.append(tmp)

                formulas, formulas_length = pad_batch_formulas(formulas_list, self.vocab.id_pad, self.vocab.id_end)
                formulas = torch.LongTensor(formulas).to(self.device)
                formulas = torch.unsqueeze(formulas, dim=2)
                self.formulas = formulas
                # self.formulas_length = torch.IntTensor(formulas_length).to(self.device)
            else:
                self.batch_images = torch.FloatTensor(input[0]).to(self.device)
                formulas_list = []
                for i in input[1]:
                    tmp = list(self._form_prepro(i))
                    # 公式str预处理转int的label时, 在label 前插入init 的label
                    tmp.insert(0, self.vocab.id_init)
                    formulas_list.append(tmp)

                formulas, formulas_length = pad_batch_formulas(formulas_list, self.vocab.id_pad, self.vocab.id_end)
                formulas = torch.LongTensor(formulas).to(self.device)
                formulas = torch.unsqueeze(formulas, dim=2)
                self.formulas = formulas
                # self.formulas_length = torch.IntTensor(formulas_length).to(device)
        # 测试时从此通过
        else:
            self.batch_images = torch.FloatTensor(input[0]).to(self.device)
            formulas_list = []
            for i in input[1]:
                tmp = list(self._form_prepro(i))
                # 公式str预处理转int的label时, 在label 前插入init 的label
                tmp.insert(0, self.vocab.id_init)
                formulas_list.append(tmp)
            self.formulas = formulas_list

    def _decode_and_generate(self, decoder_in, memory_bank, memory_lengths, step=None):
        dec_out, dec_attn = self.net.decoder(decoder_in, memory_bank, memory_lengths=memory_lengths, step=step)
        attn = dec_attn["std"]
        log_probs = self.net.generator(dec_out.squeeze(0))
        return log_probs, attn

    def forward(self):
        if self.args.isTrain:
            self.logits, self.attn = self.net(self.batch_images, self.formulas, None)
        else:
            bs = self.batch_images.shape[0]
            enc_states, memory_bank, src_lengths = self.net.encoder(self.batch_images, None)
            if src_lengths is None:
                src_lengths = torch.Tensor(bs).long().fill_(memory_bank.shape[0]).to(memory_bank.device)
            self.net.decoder.init_state(self.batch_images, memory_bank, enc_states)
            results = {
                "predictions": None,
                "scores": None,
                "attention": None,
                "batch": self.batch_images,
                "gold_score": [0] * bs}
            # src_map = None
            self.net.decoder.map_state(lambda state, dim: tile(state, self.beam_size, dim=dim))
            memory_bank = tile(memory_bank, self.beam_size, dim=1)
            mb_device = memory_bank.device
            memory_lengths = tile(src_lengths, self.beam_size)
            beam = BeamSearch(
                self.beam_size,
                n_best=self.n_best,
                batch_size=bs,
                global_scorer=self.global_scorer,
                pad=self.vocab.id_pad,
                eos=self.vocab.id_end,
                bos=self.vocab.id_init,
                min_length=0,
                ratio=-0.0,
                max_length=self.max_length,
                mb_device=mb_device,
                return_attention=False,
                stepwise_penalty=False,
                block_ngram_repeat=0,
                exclusion_tokens=self._exclusion_idxs,
                memory_lengths=memory_lengths)
            for step in range(self.max_length):
                decoder_input = beam.current_predictions.view(1, -1, 1)
                log_probs, attn = self._decode_and_generate(
                    decoder_input,
                    memory_bank,
                    memory_lengths=memory_lengths,
                    step=step)
                beam.advance(log_probs, attn)
                any_beam_is_finished = beam.is_finished.any()
                if any_beam_is_finished:
                    beam.update_finished()
                    if beam.done:
                        break

                select_indices = beam.current_origin

                if any_beam_is_finished:
                    # Reorder states.
                    if isinstance(memory_bank, tuple):
                        memory_bank = tuple(x.index_select(1, select_indices)
                                            for x in memory_bank)
                    else:
                        memory_bank = memory_bank.index_select(1, select_indices)

                    memory_lengths = memory_lengths.index_select(0, select_indices)
                self.net.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))
            results["scores"] = beam.scores
            results["predictions"] = beam.predictions
            results["attention"] = beam.attention
            self.results = results

    def preprocess(self):
        ratio = 1
        input = self.input[0] if self.vis_flag else self.input
        self.src_img_shape = input.shape
        self.ratio = ratio
        self.src_img = input[np.newaxis, :, :]
        self.preprocess_img = self.src_img

    # def infer_batch(self, vis_path=None, vis_flag=None):
    #     self.vis_flag = vis_flag
    #     self.preprocess()
    #     self.set_input(self.preprocess_img)
    #     self.forward()
    #     out = self.postprocess()
    #     if vis_flag:
    #         self.vis_img_batch(out, vis_flag)

    # def vis_img(self, out=None, vis_path=None):
    #     """
    #     可视化测试结果
    #     :param im: im (numpy, (h, w, c)
    #     :param cords: cords list[n tuple]   (xmin, ymin, xmax, ymax)
    #     :param save_path: 可视化保存路径
    #     :return:
    #     """
    #     # img = self.input[0].copy()
    #     # new_img = 255 * np.ones((2 * img.shape[0], 2 * img.shape[1]), dtype=np.uint8)
    #     # new_img[:img.shape[0], :img.shape[1]] = img
    #     # cv2.putText(new_img, str(out[0][0]), (int(0.05 * img.shape[1]), img.shape[0]), \
    #     #             cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 1)
    #     #
    #     # img_name = os.path.join(vis_path, self.input[1] + ".png")
    #     # # print(img_name, out[0])
    #     # # gt_file = os.path.join(vis_path, self.input[1] + ".txt")
    #     # #
    #     # # score = 0.9
    #     #
    #     # cv2.imwrite(img_name, new_img)
    #     print(out)
    #     img_abs_name = os.path.join("/media/chen/wt/pytorch-dl-framework/out/images", self.input[1] + ".png")
    #     visual_one_res(img_abs_name, out[0][0], vis_path)
    #
    # def vis_img_batch(self, out=None, vis_path=None):
    #     for i in range(self.preprocess_img.shape[0]):
    #         img_abs_name = os.path.join("/media/Data/hzc/datasets/formulas/all_test", self.input[i][1] + ".png")
    #         visual_one_res(img_abs_name, out[i][0], vis_path)

    def cal_loss(self):
        self.forward()
        recog_loss, n_words, n_correct = self.criterion(self.formulas, self.logits, self.attn, \
                                                        normalization=self.opt.batch_size, \
                                                        trunc_size=self.formulas.shape[1])
        # recog_loss_sum = sum(recog_loss)
        # n_words_sum = sum(n_words)
        # n_correct_sum = sum(n_correct)
        self.loss = recog_loss
        self.xent_loss = recog_loss / n_words if n_words != 0 else recog_loss
        # self.acc = torch.FloatTensor([float(100 * n_correct / n_words)]).to(recog_loss.device)
        # self.ppl = torch.FloatTensor([float(math.exp(min(recog_loss / n_words, 100)))]).to(recog_loss.device)
        # self.xent = recog_loss / n_correct

