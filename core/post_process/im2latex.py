# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/20

from core.modules.beam_search import Translation
from core.modules.text import Vocab
from . import register_post_process


@register_post_process
class Im2latex_postprocess(object):
    def __init__(self, vocab_cfg, n_best=1, replace_unk=False):
        self.n_best = n_best
        self.vocab = Vocab(vocab_cfg)
        self.vocab_size = self.vocab.n_tok
        self.replace_unk = replace_unk
        self.phrase_table = ""

    def __call__(self, outputs, img):
        output = outputs[0]

        translations = self.from_batch(output)
        pred_score_total, pred_words_total = 0, 0
        all_scores = []
        all_predictions = []

        for trans in translations:
            all_scores += [trans.pred_scores[:self.n_best]]
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:self.n_best]]
            all_predictions += [n_best_preds]
        return all_predictions

    def build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        # tgt_field = dict(self.fields)["tgt"].base_field
        # vocab = tgt_field.vocab
        tokens = []
        # pred 丢失了第一位
        pred = pred.cpu().numpy().tolist()
        for tok in pred:
            if tok < self.vocab_size:
                tokens.append(self.vocab.id_to_tok[tok])
            # else:
            #     tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == self.vocab.special_tokens[3]:
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == self.vocab.special_tokens[0]:
                    _, max_index = attn[i][:len(src_raw)].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table != "":
                        with open(self.phrase_table, "r") as f:
                            for line in f:
                                if line.startswith(src_raw[max_index.item()]):
                                    tokens[i] = line.split('|||')[1].strip()
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]

        batch_size = batch.shape[0]

        preds, pred_score, attn, gold_score = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"]), key=lambda x: x[-1])))

        # Sorting
        src = None
        translations = []
        for b in range(batch_size):
            src_vocab = None
            src_raw = None
            pred_sents = [self.build_target_tokens(
                None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            gold_sent = None

            translation = Translation(
                src[:, b] if src is not None else None,
                src_raw, pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)
        return translations


