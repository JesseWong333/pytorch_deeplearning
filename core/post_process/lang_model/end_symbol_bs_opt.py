# encoding: utf-8
import numpy as np
import jieba

'''for server'''
from .de_symbol import desymbol

# from de_symbol import desymbol

symbol_list = "',.;:\\|/?\"`^&*()<>{}[]!@#$%*-_+=，。；：“”‘’｛｝【】、《》？！￥……（）—~·〔 〕 ˉ¯▲"
# 句尾最后一个标点，成对符号或者结尾标点
first_pair_symbol_list = "',.;:?()<>{}[]!，。；：“”‘’\"'｛｝【】、《》？！……（）—〔 〕 ▲"

# 全半角等价处理列表
equal_sym_key_list = [['(', '（'], [')', '）']]
equal_sym_value_list = ['(', ')']
# 成对符号及其替换项目列表
exp_list = [['()', '（）', '(>', '（>', '<)', '<）', '(）', '（)'], ['[]'], ['【】', '[】', '【]', '[〕', '〔]', '〔】', '【〕'],
            ['<>'], ['(▲)', '（▲）', '（▲)', '(▲）']]
replace_list = ['()', '[]', '【】', '<>', '（▲）']

equal_sym_dict = {}
exp_replace_dict = {}

# for resnet
only_head_syms_2add = '(（[【）)]】'
tail_sym_not2add = '-¯ˉ~―`^“”='

for keys, val in zip(equal_sym_key_list, equal_sym_value_list):
    for k in keys:
        equal_sym_dict[k] = val

for exps, replace_item in zip(exp_list, replace_list):
    for exp in exps:
        exp_replace_dict[exp] = replace_item


# 字符串是否包含成对符号
def is_contain_pair_sym(k):
    for sym_pair in exp_replace_dict:
        if sym_pair in k:
            return sym_pair
    return None


# 获取两端标点字符串
def get_end_symbol(string, type='tail'):
    ret = ''
    if type == 'tail':
        search_string = string[::-1]
    else:
        search_string = string

    for c in search_string:
        if c not in symbol_list:
            break
        else:
            # if c!=' ':
            ret += c
    return ret if type != 'tail' else ret[::-1]


# 获取首个配对标点及其所在字符串中的位置
def get_first_pair_symbol(string, type='tail', blank_step=2):
    ret = ''
    ret_id = -1
    if type == 'tail':
        search_string = string[::-1]
    else:
        search_string = string

    for i, c in enumerate(search_string):
        if c not in first_pair_symbol_list:
            break
        else:
            if c in only_head_syms_2add:
                ret = c
                ret_id = i
                break

    return (ret, ret_id) if type != 'tail' else (ret, len(string) - 1 - ret_id)


# 删除两端标点符号
def de_end_symbol(string, type='tail'):
    ret = ''
    if type == 'tail':
        search_string = string[::-1]
    else:
        search_string = string

    for i, c in enumerate(search_string):
        if c not in symbol_list:
            ret = search_string[i:]
            break

    return ret if type != 'tail' else ret[::-1]


def deRepeatInSeq(sym):
    ret = ''
    # 统一等价标点
    for c in sym:
        if c in equal_sym_dict:
            ret += equal_sym_dict[c]
        else:
            ret += c

    # 对连续相同标点去重
    tmp = ''
    prev_char = ''
    for c in ret:
        if c != prev_char:
            if c != ' ':
                prev_char = c
            tmp += c

    return tmp


def replace_candidate_with_sym(candidates, sym, type='tail'):
    replace_sym_candidates = []

    # 去除重复标点,特别是半圆括号
    sym = deRepeatInSeq(sym)

    for candidate in candidates:
        de_sym_candidate = de_end_symbol(candidate, type)
        if type == 'tail':
            replace_sym_candidates.append(de_sym_candidate + sym)
        else:
            replace_sym_candidates.append(sym + de_sym_candidate)

    return replace_sym_candidates


def get_prior_replace_item_from_list(iter):
    prior_replace_item = ''
    expect_sym_pair_key = ''
    for k in iter:  # end_sym_analysis.keys():
        '''
        配对字符优先级别最高，以替换方式进行
        '''
        sym_may2expect_key = is_contain_pair_sym(k)
        if sym_may2expect_key is not None:
            # 选择最长的优先替换项，即包含。等标点,后续替换时以去重保证一定正确性
            if len(k) > len(prior_replace_item):
                prior_replace_item = k
                expect_sym_pair_key = sym_may2expect_key
    return prior_replace_item, expect_sym_pair_key


def replace_string_by_id(string, begin=0, end=-1, replace_item=None):
    '''
    begin==end: insert
    begin<end: replace
    begin>end: no
    :param string handle
    :param begin:
    :param end:
    :param replace_item:
    :return:
    '''
    if (not string) or (not len(string)) or begin > end or (not replace_item):
        return string

    return string[:begin] + replace_item + string[end:]


def find_sym2add(candidate=None, type='tail'):
    if (candidate is None) or (len(candidate) < 2):
        return candidate

    is_len_satified = False
    max_len = 0
    min_len = 1000
    is_all_pair_sym = True

    for string in candidate:
        no_blank_str = string.strip(' ')
        string_len_no_blank = len(no_blank_str)
        max_len = max(max_len, string_len_no_blank)
        min_len = min(min_len, string_len_no_blank)
        # 长度为1且为配对符号时才补充
        # is_all_pair_sym = is_all_pair_sym and (len(no_blank_str)==1 and  no_blank_str in only_head_syms_2add)

        if string_len_no_blank > 1:
            is_len_satified = True

    # 单字符且去空格长度max=1,min=0
    end_sym_analysis4single_sym = {}
    candidate_id_no_tail_sym = []
    if max_len == 1 and min_len == 0:
        for id, string in enumerate(candidate):
            no_blank_str = string.strip(' ')
            if no_blank_str in equal_sym_dict:
                no_blank_str = equal_sym_dict[no_blank_str]

            if no_blank_str in equal_sym_dict:
                no_blank_str = equal_sym_dict[no_blank_str]

            if no_blank_str:
                num_order = end_sym_analysis4single_sym.get(no_blank_str, None)
                if num_order:
                    end_sym_analysis4single_sym[no_blank_str] = (num_order[1] + 1, id)
                else:
                    end_sym_analysis4single_sym[no_blank_str] = (1, id)
            else:
                candidate_id_no_tail_sym.append(id)
        end_sym_analysis4single_sym = sorted(end_sym_analysis4single_sym.items(), key=lambda x: (x[1][0], -x[1][1]),
                                             reverse=True)

        symbol2add = end_sym_analysis4single_sym[0][0] if len(end_sym_analysis4single_sym) > 0 else ''

        for i in candidate_id_no_tail_sym:
            candidate[i] += symbol2add

        return candidate

    # 候选集合中字符串长度不超过1
    if not is_len_satified:
        return candidate

    # 长度大于2
    # [(symstr,id),...]
    candidate_end_symstr_list = []
    candidate_id_no_tail_sym = []
    end_sym_analysis = {}
    for i, string in enumerate(candidate):
        end_sym = get_end_symbol(string, type)
        candidate_end_symstr_list.append(end_sym)

        no_blank_end_sym = end_sym.replace(' ', '')
        if no_blank_end_sym:
            num_order = end_sym_analysis.get(no_blank_end_sym, None)
            if num_order is None:
                end_sym_analysis[no_blank_end_sym] = (1, i)
            else:
                end_sym_analysis[no_blank_end_sym] = (num_order[0] + 1, i)
        else:
            candidate_id_no_tail_sym.append(i)

    # '''' for debug'''
    # print(end_sym_analysis)
    # print(candidate_end_symstr_list)

    # 当候选集合中的配对符号左右分别出现在不同候选中时，
    # 搜索可能的配对符号
    blank_step = 2
    first_pair_symbol_list = []
    combine_pair_symbol = set()
    # [[isToRef,symIdInStr],...]
    first_symid_ref_cache = []
    for symstr in candidate_end_symstr_list:
        first_pair_symbol, first_sym_id = get_first_pair_symbol(symstr, type)
        # first_pair_symbol.append(get_first_pair_symbol(k)[0])
        isToRef = False

        # for debug
        # print(symstr,first_pair_symbol,first_sym_id)

        if type == 'tail':
            if first_pair_symbol in only_head_syms_2add[len(only_head_syms_2add) // 2:]:
                prev_str = symstr[max(first_sym_id - blank_step, 0):first_sym_id]
                # 特殊情况如(哈)，右配对括号必须满足以下任一条件
                if len(prev_str) == blank_step or len(prev_str.strip(' ')) > 0:
                    first_pair_symbol_list.append(first_pair_symbol)
                    isToRef = True
            elif first_pair_symbol in only_head_syms_2add[:len(only_head_syms_2add) // 2]:
                first_pair_symbol_list.append(first_pair_symbol)
                isToRef = True
        elif type == 'head':
            if first_pair_symbol in only_head_syms_2add[:len(only_head_syms_2add) // 2]:
                next_str = symstr[first_sym_id + 1:i + 1 + blank_step]
                # 特殊情况如(哈)，左配对括号必须满足以下任一条件
                if len(next_str) == blank_step or len(next_str.strip(' ')) > 0:
                    first_pair_symbol_list.append(first_pair_symbol)
                    isToRef = True
            elif first_pair_symbol in only_head_syms_2add[len(only_head_syms_2add) // 2:]:
                first_pair_symbol_list.append(first_pair_symbol)
                isToRef = True
        first_symid_ref_cache.append([isToRef, first_sym_id])

    # 将配对符号排序在相邻位置
    sorted_first_pair_symbol = sorted(first_pair_symbol_list)
    # 找出配对符号并转化为统一格式
    for i in range(0, len(sorted_first_pair_symbol) - 1):
        may_pair_sym = sorted_first_pair_symbol[i] + sorted_first_pair_symbol[i + 1]
        if may_pair_sym in exp_replace_dict:
            combine_pair_symbol.add(exp_replace_dict[may_pair_sym])

    # print(type, 'sorted_first_pair_symbol:',sorted_first_pair_symbol)
    # print(type, 'combine_pair_symbol:', combine_pair_symbol)

    # 从统计标点字典中查找优先替换项
    prior_replace_item, expect_sym_pair_key = get_prior_replace_item_from_list(end_sym_analysis.keys())

    # 优先替换项失效，搜索组合优先替换项
    if not prior_replace_item and len(combine_pair_symbol):
        # print('组合优先替换项')
        combine_pair_symbol = list(combine_pair_symbol)

        for i, (isToRef, symIdInStr) in enumerate(first_symid_ref_cache):
            if isToRef:
                candidate_end_symstr_list[i] = replace_string_by_id(candidate_end_symstr_list[i], symIdInStr,
                                                                    symIdInStr + 1, combine_pair_symbol[0])
        for i, string in enumerate(candidate_end_symstr_list):
            candidate_end_symstr_list[i] = string.replace(' ', '')

        prior_replace_item, expect_sym_pair_key = get_prior_replace_item_from_list(candidate_end_symstr_list)

    # 优先替换项
    if prior_replace_item:
        # print('优先替换项')
        expect_sym_pair_value = exp_replace_dict[expect_sym_pair_key]
        expect_sym_pair_value = expect_sym_pair_value[0] + '  ' + expect_sym_pair_value[1] if len(
            expect_sym_pair_value) == 2 else expect_sym_pair_value
        prior_replace_item = prior_replace_item.replace(expect_sym_pair_key, expect_sym_pair_value)

        # for debug
        # print(expect_sym_pair_key, expect_sym_pair_value, prior_replace_item)

        return replace_candidate_with_sym(candidate, prior_replace_item, type=type)

    else:
        # print('末位追加')
        end_sym_analysis = sorted(end_sym_analysis.items(), key=lambda x: (x[1][0], -x[1][1]), reverse=True)

        symbol2add = end_sym_analysis[0][0] if len(end_sym_analysis) > 0 else ''

        if symbol2add:
            if type == 'tail':
                if symbol2add not in tail_sym_not2add:
                    for i in candidate_id_no_tail_sym:
                        candidate[i] = candidate[i].strip(' ') + symbol2add
            else:
                if symbol2add in only_head_syms_2add:  # '''for resnet'''
                    for i in candidate_id_no_tail_sym:
                        candidate[i] = symbol2add + candidate[i].strip(' ')
    return candidate


def add_sym2candidate(candidate=None):
    if (candidate is None) or (not len(candidate)):
        return candidate

    candidate = find_sym2add(candidate, type='tail')
    # print('before：',candidate)

    candidate = find_sym2add(candidate, type='head')
    # print('after：',candidate)

    return candidate


if __name__ == '__main__':

    test_example_list = [
        ['全球共分几大板块 （', '全球共分几大板块）', '全球共分几大板块）', '全球共分几大板块', '全球共分几大板块（.'],
        # 配对符号不成对出现于候选中，且不符合组合配对的条件0>=black_step
        ['全球共分几大板块（', '全球共分几大板块 ）', '全球共分几大板块）', '全球共分几大板块', '全球共分几大板块（.'],
        # 配对符号不成对出现于候选中，且不符合组合配对的条件1>=black_step
        ['全球共分几大板块（', '全球共分几大板块  ）', '全球共分几大板块）', '全球共分几大板块', '全球共分几大板块（.'],
        # 配对符号不成对出现于候选中，符合组合配对的条件2>=black_step
        ['全球共分几大板块（', '全球共分几大板块。）', '全球共分几大板块）', '全球共分几大板块', '全球共分几大板块（.'],  # 配对符号不成对出现于候选中，符合组合配对的条件black_step字符串中存在标点
        ['全球共分几大板块 （', '全球共分几大板块（）', '全球共分几大板块）', '全球共分几大板块', '全球共分几大板块（）.'],  # 配对符号成对出现于候选中，
        ['全球共分几大板块 （', '全球共分几大板块。（）', '全球共分几大板块）', '全球共分几大板块', '全球共分几大板块（）.'],
        ['(1)粒子进入？', '1)粒子进入', '（1）粒子进入?', '(1)粒子进入'],
        ['全球共分几大板块 （', '全球共分几大板块（）.', '全球共分几大板块）.', '全球共分几大板块', '全球共分几大板块(.)'],

        [')粒子进入', '（粒子进入', '（粒子进入', '）粒子进入', '(粒子进入'],  # 配对符号不成对出现于候选中，且不符合组合配对的条件0>=black_step
        [')粒子进入', '（粒子进入', '（粒子进入', '）粒子进入', '( 粒子进入'],  # 配对符号不成对出现于候选中，且不符合组合配对的条件1>=black_step
        [')粒子进入', '（粒子进入', '（粒子进入', '）粒子进入', '(  粒子进入'],  # 配对符号不成对出现于候选中，且不符合组合配对的条件2>=black_step
        [')粒子进入', '（粒子进入', '（粒子进入', '）粒子进入', '(.粒子进入'],  # 配对符号不成对出现于候选中，符合组合配对的条件black_step字符串中存在标点
        [')粒子进入', '（粒子进入', '（粒子进入', '）粒子进入', '( )粒子进入'],
        [')粒子进入', '（粒子进入', '（粒子进入', '）粒子进入', '(）粒子进入'],
        [')粒子进入', '（粒子进入', '（粒子进入', '）粒子进入', '(> 粒子进入'],

        ['序是（  ）', '序是（   ）', '序是（  ））', '序是（    ）', '序是（  ） ）'],  # 配对符号成对出现，标点列表冗余

        ['人口出生率(%)', '人口出生率(%', '人口出生率(%）', '人口出生率(9%)', '人口出生率(%]'],  # 配对符号，中间出现特殊符号情况

        ['[', '[】', '[I', '[〕', '[J'],  # 可能单侧识别错误的可能配对符号，替换
        ['[', '[〕', '[】', '[I', '[J'],  # 可能单侧识别错误的可能配对符号，替换

        ['performance', 'performance.', 'performance,', 'performancec', 'perfornmance'],  # 末尾句点
        ['C. A manager ', 'C. A manager', 'C. A manager.', 'C.A manager ', 'C. A manager. '],  # 末尾句点，带空格情况

        ['）', '﹚', '〕', ')', ''],  # 单标点
        ['﹚', '）', '', '〕', ' ﹚'],

        # ['_A是', '_A是_', '-A是', 'A是', '_4是'], # 句首标点追加特殊情况
        # ['A．冠状病毒', 'A．冠状病毒ˉ', 'A．冠状病毒¯', 'A．冠状病毒―', 'A．冠状病毒~'], #末尾不追加标点
        ['1．下列图案由正多边形拼成，其中既是轴对称图形又是中心对称图形的是（▲）', '1．下列图案由正多边形拼成，其中既是轴对称图形又是中心对称图形的是（▲',
         '1．下列图案由正多边形拼成，其中既是轴对称图形又是中心对称图形的是（）', '1．下列图案由正多边形拼成，其中既是轴对称图形又是中心对称图形的是（ ）',
         '1．下列图案由正多边形拼成，其中既是轴对称图形又是中心对称图形的是（'],
    ]
    test_sample4deRepeat = [') ()', '.().', '。（().']
    # for sample in test_sample4deRepeat:
    #     print(deRepeatInSeq(sample))
    #
    # exit()

    for bs_candidates in test_example_list[-1:]:
        print(bs_candidates)
        bs_candidates = add_sym2candidate(bs_candidates)

        bs_batch_candidate_ppl = np.array(
            [len(list(jieba.cut(desymbol(candidate), HMM=False))) for candidate in bs_candidates])
        min_ppl_id = np.argmin(bs_batch_candidate_ppl)
        if bs_batch_candidate_ppl[0] <= bs_batch_candidate_ppl[min_ppl_id]:
            min_ppl_id = 0
        # for debug
        print(bs_candidates, bs_batch_candidate_ppl, min_ppl_id)
        print()
    # print(equal_sym_dict)
    # print(exp_replace_dict)
