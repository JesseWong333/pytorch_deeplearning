import numpy as np
import jieba
'''for server'''
from .de_symbol import desymbol,symbol_list
# from de_symbol import desymbol,symbol_list

order_symbol_list = '0123456789' \
    'abcdefghijklmnopqrstuvwxyz' \
          'ABCDEFGHIJKLMNOPQRSTUVWXYZ'\
          ' ' #.

def replace_string_by_id(string,begin=0,end=-1,replace_item=None):
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
    if (not string) or (not len(string)) or begin>end or (not replace_item):
        return string

    return string[:begin]+replace_item+string[end:]

# 获取两端第一个存在于order_symbol_list 的字符
def get_end_symbol(string,type='head'):
    ret = ''
    ret_id = -1

    if type=='tail':
        search_string = string[::-1]
    else:
        search_string = string

    for id,c in enumerate(search_string):
        if c in symbol_list:
            continue
        elif c not in order_symbol_list :
            break
        else:
            if c!=' ':
                ret = c
                ret_id = id
                break

    return ret,ret_id

def correct(candidate):
    candidate_id_to_add = []
    candidate_id_to_replace = []
    end_sym_analysis = {}

    for i, string in enumerate(candidate):
        end_sym, sym_id = get_end_symbol(string)

        if end_sym:
            num_order = end_sym_analysis.get(end_sym, None)
            if num_order is None:
                end_sym_analysis[end_sym] = (1, i)
            else:
                end_sym_analysis[end_sym] = (num_order[0] + 1, i)
            candidate_id_to_replace.append([i, sym_id])
        else:
            candidate_id_to_add.append([i, sym_id])

    # print(end_sym_analysis)

    end_sym_analysis = sorted(end_sym_analysis.items(), key=lambda x: (x[1][0], -x[1][1]), reverse=True)

    symbol2add = end_sym_analysis[0][0] if len(end_sym_analysis) > 0 else ''

    # 是否大写化所有选项序号
    # 当大写字母数量大于等于小写数时为True
    is_upper_all = False

    # 大小写数量相同优选大写
    if str.islower(symbol2add):
        end_sym_analysis = dict(end_sym_analysis)
        upper_item = end_sym_analysis.get(str.upper(symbol2add), None)
        # 大写化所有选项序号的力度
        # if upper_item is not None and upper_item[0] == end_sym_analysis[symbol2add][0]:
        if upper_item is not None:
            symbol2add = str.upper(symbol2add)
            is_upper_all = True
    if str.isupper(symbol2add):
        is_upper_all = True

    '''for bug'''
    # print('symbol2add: ', symbol2add)
    # print('candidate_id_to_add: ', candidate_id_to_add)
    # print('candidate_id_to_replace: ', candidate_id_to_replace)

    # add_args:[candidate_id, sym_id_in_string]
    for add_args in candidate_id_to_add:
        candidate_id, sym_id_in_string = add_args
        candidate[candidate_id] = symbol2add + candidate[candidate_id]

    # print('after add:',candidate)

    for replace_args in candidate_id_to_replace:
        candidate_id, sym_id_in_string = replace_args
        if is_upper_all:
            candidate[candidate_id] = replace_string_by_id(candidate[candidate_id], sym_id_in_string,
                                                           sym_id_in_string + 1, symbol2add)
            # 特殊情况: A
            spec_replaced_item = symbol2add+'…'
            spec_to_replace_item = symbol2add + '.'
            candidate[candidate_id] = candidate[candidate_id].replace(spec_replaced_item,spec_to_replace_item)

    # print(candidate)
    return candidate

if __name__ == '__main__':
    test_example_list = [
         ['A.学习行为', 'a.学习行为', '..学习行为', '·.学习行为', 'A学习行为'],
         ['f.学习行为', 'a.学习行为', 'f..学习行为', '·.学习行为', 'A学习行为'],
         ['a．细菌繁殖的后代很多，抗生素用量不够', 'A．细菌繁殖的后代很多，抗生素用量不够', '．细菌繁殖的后代很多，抗生素用量不够', 'á．细菌繁殖的后代很多，抗生素用量不够',
          '·．细菌繁殖的后代很多，抗生素用量不够'],
         ['A.学习行为', 'a.学习行为', '..学习行为', '·.学习行为', 'A学习行为'],
         ['A．甲→乙→丙→丁', 'A．甲→乙→丙+丁', 'a．甲→乙→丙→丁', '·．甲→乙→丙→丁', '．甲→乙→丙→丁'],
         ['3.右图表示的是一个生态系统中某些生物的相对数量关系，这些生物构成了一条食物链。这条食物链中物', '3.右图表示的是一个生态系统中某些生物的相对数量关系，这些生物构成了一条食物链。这条食物辩中物',
         'a.右图表示的是一个生态系统中某些生物的相对数量关系，这些生物构成了一条食物链。这条食物链中物', '..右图表示的是一个生态系统中某些生物的相对数量关系，这些生物构成了一条食物链。这条食物链中物',
         '.右图表示的是一个生态系统中某些生物的相对数量关系，这些生物构成了一条食物链。这条食物链中物'],
        ['11. What time will the speakers go to the National Park tomorrow morning?',
         '11. What time will the speakers go to the National Park tomorrow morning? ',
         '1l. What time will the speakers go to the National Park tomorrow morning?',
         'l1. What time will the speakers go to the National Park tomorrow morning?',
         '11. What  time will the speakers go to the National Park tomorrow morning?'],
        ['_A是', '_A是_', '-A是', 'A是', '_4是']  # 句首标点追加特殊情况
    ]

    # tests = ['1','rep']
    # for t in tests:
    #     print(replace_string_by_id(t,0,0,'ok'))
    #     print(replace_string_by_id(t, 0, 1, 'ok'))
    #     print(replace_string_by_id(t, 1, 0, 'ok'))
    #
    # exit()

    for candidate in test_example_list:
    # candidate = test_example_list[0]
        print(candidate)
        candidate = correct(candidate)
        print(candidate)

        bs_batch_candidate_ppl = np.array(
            [len(list(jieba.cut(desymbol(string), HMM=False))) for string in candidate])
        min_ppl_id = np.argmin(bs_batch_candidate_ppl)
        if bs_batch_candidate_ppl[0] <= bs_batch_candidate_ppl[min_ppl_id]:
            min_ppl_id = 0
        # for debug
        print(candidate, bs_batch_candidate_ppl, min_ppl_id)