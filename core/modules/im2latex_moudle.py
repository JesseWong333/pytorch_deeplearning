# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/20

import numpy as np


def pad_batch_formulas(formulas, id_pad, id_end, max_len=None):
    """Pad formulas to the max length with id_pad and adds and id_end token
    at the end of each formula

    Args:
        formulas: (list) of list of ints
        max_length: length maximal of formulas

    Returns:
        array: of shape = (batch_size, max_len) of type np.int32
        array: of shape = (batch_size) of type np.int32

    """
    if max_len is None:
        max_len = max(list(map(lambda x: len(x), formulas)))

    batch_formulas = id_pad * np.ones([len(formulas), max_len+1],
            dtype=np.int32)
    formula_length = np.zeros(len(formulas), dtype=np.int32)
    for idx, formula in enumerate(formulas):
        batch_formulas[idx, :len(formula)] = np.asarray(formula,
                dtype=np.int32)
        batch_formulas[idx, len(formula)] = id_end
        formula_length[idx] = len(formula) + 1

    return batch_formulas, formula_length