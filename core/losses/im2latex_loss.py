# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging
import torch
import time
import torch.nn as nn
import math
import sys
from . import register_loss

#
# logger = logging.getLogger()
#
#
# def init_logger(log_file=None, log_file_level=logging.NOTSET):
#     log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(log_format)
#     logger.handlers = [console_handler]
#
#     if log_file and log_file != '':
#         file_handler = logging.FileHandler(log_file)
#         file_handler.setLevel(log_file_level)
#         file_handler.setFormatter(log_format)
#         logger.addHandler(file_handler)
#
#     return logger


#
# class Im2LatexLoss(nn.Module):
#
#     def __init__(self):
#         super(Im2LatexLoss, self).__init__()
#         self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
#
#     def gen_binary_mask(self, formulas_length):
#         max_len = torch.max(formulas_length).cpu().numpy()
#         num = formulas_length.shape[0]
#         mask = torch.zeros(num, max_len).byte().to(formulas_length.device)
#         for i in range(num):
#             mask[i, :formulas_length[i]] = 1
#         return mask
#
#     def forward(self, logits, formulas, formulas_length):
#         # traing cal loss
#         logits = logits.permute(0, 2, 1)
#         # print(logits.shape, formulas.shape, formulas_length.shape, logits.dtype, formulas.dtype)
#         losses = self.cross_entropy(logits, formulas)
#         mask = self.gen_binary_mask(formulas_length)
#         losses = torch.masked_select(losses, mask)
#         loss = losses.mean()
#         # compute perplexity for test
#         ce_words = losses.sum() # sum of CE for each word
#         n_words = formulas_length.sum() # number of words
#         return loss


def filter_shard_state(state, shard_size=None):
    # print("filter", shard_size)
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            # print("for", k, state[k].requires_grad, type(v), type(v_split))

            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                # print("for1", k, state[k].requires_grad)
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
            # for v_chunk in v_split:
            #     print("grad, v_chunk", v_chunk, v_chunk.grad, type(v_chunk), v_chunk.requires_grad)
        # print("variables", variables)
        inputs, grads = zip(*variables)
        # print(grads)
        # print("state", state.keys())
        # print(state["output"].shape, state["output"].dtype, state["output"].requires_grad)
        # print(state["target"].shape, state["target"].dtype, state["target"].requires_grad)
        # print("autograd", type(inputs), type(grads), type(state))
        # for i in inputs:
        #     print("i", i.shape)
        # for j in grads:
        #     print("j", j.shape)
        torch.autograd.backward(inputs, grads)

#
# class Statistics(object):
#     """
#     Accumulator for loss statistics.
#     Currently calculates:
#
#     * accuracy
#     * perplexity
#     * elapsed time
#     """
#
#     def __init__(self, loss=0, n_words=0, n_correct=0):
#         self.loss = loss
#         self.n_words = n_words
#         self.n_correct = n_correct
#         self.n_src_words = 0
#         self.start_time = time.time()
#
#     @staticmethod
#     def all_gather_stats(stat, max_size=4096):
#         """
#         Gather a `Statistics` object accross multiple process/nodes
#
#         Args:
#             stat(:obj:Statistics): the statistics object to gather
#                 accross all processes/nodes
#             max_size(int): max buffer size to use
#
#         Returns:
#             `Statistics`, the update stats object
#         """
#         stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
#         return stats[0]
#
#     @staticmethod
#     def all_gather_stats_list(stat_list, max_size=4096):
#         """
#         Gather a `Statistics` list accross all processes/nodes
#
#         Args:
#             stat_list(list([`Statistics`])): list of statistics objects to
#                 gather accross all processes/nodes
#             max_size(int): max buffer size to use
#
#         Returns:
#             our_stats(list([`Statistics`])): list of updated stats
#         """
#         from torch.distributed import get_rank
#         from onmt.utils.distributed import all_gather_list
#
#         # Get a list of world_size lists with len(stat_list) Statistics objects
#         all_stats = all_gather_list(stat_list, max_size=max_size)
#
#         our_rank = get_rank()
#         our_stats = all_stats[our_rank]
#         for other_rank, stats in enumerate(all_stats):
#             if other_rank == our_rank:
#                 continue
#             for i, stat in enumerate(stats):
#                 our_stats[i].update(stat, update_n_src_words=True)
#         return our_stats
#
#     def update(self, stat, update_n_src_words=False):
#         """
#         Update statistics by suming values with another `Statistics` object
#
#         Args:
#             stat: another statistic object
#             update_n_src_words(bool): whether to update (sum) `n_src_words`
#                 or not
#
#         """
#         self.loss += stat.loss
#         self.n_words += stat.n_words
#         self.n_correct += stat.n_correct
#
#         # if update_n_src_words:
#         #     self.n_src_words += stat.n_src_words
#
#     def accuracy(self):
#         """ compute accuracy """
#         return 100 * (self.n_correct / self.n_words)
#
#     def xent(self):
#         """ compute cross entropy """
#         return self.loss / self.n_words
#
#     def ppl(self):
#         """ compute perplexity """
#         return math.exp(min(self.loss / self.n_words, 100))
#
#     def elapsed_time(self):
#         """ compute elapsed time """
#         return time.time() - self.start_time
#
#     def output(self, step, num_steps, learning_rate, start):
#         """Write out statistics to stdout.
#
#         Args:
#            step (int): current step
#            n_batch (int): total batches
#            start (int): start time of step.
#         """
#         t = self.elapsed_time()
#         # step_fmt = "%2d" % step
#         # if num_steps > 0:
#         #     step_fmt = "%s/%5d" % (step_fmt, num_steps)
#         # logger.info(
#         #     ("Step %s; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
#         #      "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
#         #     % (step_fmt,
#         return [self.accuracy(),
#                self.ppl(),
#                self.xent(),
#                learning_rate,
#                self.n_src_words / (t + 1e-5),
#                self.n_words / (t + 1e-5)]
#
#     # def log_tensorboard(self, prefix, writer, learning_rate, step):
#     #     """ display statistics to tensorboard """
#     #     t = self.elapsed_time()
#     #     writer.add_scalar(prefix + "/xent", self.xent(), step)
#     #     writer.add_scalar(prefix + "/ppl", self.ppl(), step)
#     #     writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
#     #     writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
#     #     writer.add_scalar(prefix + "/lr", learning_rate, step)


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, ignore_index, reduction):
        super(LossComputeBase, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=ignore_index, reduction=reduction)
        # self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 formulas,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=32,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        # print("loss", trunc_start, trunc_size, shard_size)
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(formulas, output, trunc_range, attns)
        # if shard_size == 0:
        #     loss, stats = self._compute_loss(batch, **shard_state)
        #     return loss / float(normalization)
        # batch_stats = onmt.utils.Statistics()
        loss, stats = self._compute_loss(**shard_state)
        return loss / float(normalization), stats[0], stats[1]


        # losses = []
        # n_words = []
        # n_correct = []
        #
        # for shard in shards(shard_state, shard_size):
        #     # print("shard", shard)
        #     loss, stats = self._compute_loss(**shard)
        #     # loss.div(float(normalization)).backward()
        #     # 为什么要先backward, 将宽度切割成多份的意义
        #     loss.div_(float(normalization)).backward()
        #     losses.append(loss)
        #     n_words.append(stats[0])
        #     n_correct.append(stats[1])
        # return losses, n_words, n_correct

    def _stats(self, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        # print(num_non_padding, num_correct)
        return [num_non_padding, num_correct]

    def _bottle(self, _v):
        # print("_v", _v.shape)

        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

@register_loss
class Im2TextLoss(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    # def __init__(self, criterion, generator, normalization="sents",
    #              lambda_coverage=0.0):
    #     super(Im2TextLoss, self).__init__(criterion, generator)
    #     self.lambda_coverage = lambda_coverage
    #add by hcn
    def __init__(self, ignore_index, reduction, normalization="sents", lambda_coverage=0.0):
        super(Im2TextLoss, self).__init__(ignore_index, reduction)
        self.lambda_coverage = lambda_coverage


    def _make_shard_state(self, formulas, output, range_, attns=None):
        # print("s1", formulas.shape)
        if formulas.dim() == 2:
            formulas = torch.unsqueeze(formulas, 2)
        formulas = formulas.permute(1, 0, 2).contiguous()
        # print("s2", formulas.shape, range_)
        shard_state = {
            "output": output,
            "target": formulas[range_[0] + 1: range_[1], :, 0],
        }

        return shard_state

    def _compute_loss(self, output, target, std_attn=None,
                      coverage_attn=None):
        # print("s2", target.shape)

        # bottled_output = self._bottle(output)

        # scores = self.generator(bottled_output)
        gtruth = target.view(-1)
        # gtruth = target.permute(1, 0).view(-1)

        # print("scores", scores.shape, gtruth.shape, target.shape)

        loss = self.criterion(output, gtruth)

        stats = self._stats(output, gtruth)

        return loss, stats

    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum(2).view(-1)
        covloss *= self.lambda_coverage
        return covloss


# class Im2LatexLoss(nn.Module):
#
#     def __init__(self, model, tgt_field, opt, train=True):
#         super(Im2LatexLoss, self).__init__()
#         # device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")
#
#         padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
#         unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]
#         criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
#         loss_gen = model.generator
#         compute = NMTLossCompute(
#             criterion, loss_gen, lambda_coverage=opt.lambda_coverage)
#         # compute.to(device)
#
#     def forward(self, logits, formulas, formulas_length):
#         # traing cal loss
#         logits = logits.permute(0, 2, 1)
#         # print(logits.shape, formulas.shape, formulas_length.shape, logits.dtype, formulas.dtype)
#         losses = self.cross_entropy(logits, formulas)
#         mask = self.gen_binary_mask(formulas_length)
#         losses = torch.masked_select(losses, mask)
#         loss = losses.mean()
#         # compute perplexity for test
#         ce_words = losses.sum() # sum of CE for each word
#         n_words = formulas_length.sum() # number of words
#         return loss


# def build_loss_compute(model, lambda_coverage=0):
#     """
#     Returns a LossCompute subclass which wraps around an nn.Module subclass
#     (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
#     object allows this loss to be computed in shards and passes the relevant
#     data to a Statistics object which handles training/validation logging.
#     Currently, the NMTLossCompute class handles all loss computation except
#     for when using a copy mechanism.
#
#     """
#     # device = torch.device("cpu")
#     device = torch.device("cuda")
#     padding_idx = 1
#     # unk_idx = 0
#     criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
#
#     # if the loss function operates on vectors of raw logits instead of
#     # probabilities, only the first part of the generator needs to be
#     # passed to the NMTLossCompute. At the moment, the only supported
#     # loss function of this kind is the sparsemax loss.
#     loss_gen = model.generator
#
#     compute = Im2TextLoss(criterion, loss_gen, lambda_coverage=lambda_coverage)
#     compute = compute.to(device)
#
#     return compute
