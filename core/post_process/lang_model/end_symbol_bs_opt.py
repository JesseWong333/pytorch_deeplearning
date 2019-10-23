symbol_list = "',.;:\\|/?\"`^&*()<>{}[]!@#$%*-_+=，。；：“”‘’｛｝【】、《》？！￥……（）——~·〔 〕"

equal_sym_key_list= [['(','（'],[')','）']]
equal_sym_value_list =['(',')']
exp_list = [['()','（）','(>','（>','<)','<）','(）','（)'],['[]'],['【】','[】','【]','[〕','〔]','〔】','【〕'],['<>']]
replace_list = ['()','[]','【】','<>']

equal_sym_dict ={}
exp_replace_dict = {}

for keys,val in zip(equal_sym_key_list,equal_sym_value_list):
    for k in keys:
        equal_sym_dict[k] = val

for exps,replace_item in zip(exp_list,replace_list):
    for exp in exps:
        exp_replace_dict[exp]=replace_item

# 字符串是否包含成对符号
def is_contain_pair_sym(k):
    for sym_pair in exp_replace_dict:
        if sym_pair in k:
            return sym_pair
    return None

# 获取两端标点字符串
def get_end_symbol(string,type='tail'):
    ret = ''
    if type=='tail':
        search_string = string[::-1]
    else:
        search_string = string

    for c in search_string:
        if c not in symbol_list:
            break
        else:
            if c!=' ':
                ret += c
    return ret if type!='tail' else ret[::-1]

# 删除两端标点符号
def de_end_symbol(string,type='tail'):
    ret = ''
    if type=='tail':
        search_string = string[::-1]
    else:
        search_string = string

    for i,c in enumerate(search_string):
        if c not in symbol_list:
            ret = search_string[i:]
            break

    return ret if type!='tail' else ret[::-1]

def find_sym2add(candidate=None,type='tail'):
    if (candidate is None) or (len(candidate)<2):
        return None,[]

    is_len_satified=False
    for string in candidate:
        if len(string.strip(' '))>1:
            is_len_satified =True

    if not is_len_satified:
        return None,[]

    candidate_id_no_tail_sym =  []
    end_sym_analysis = {}
    for i, string in enumerate(candidate):
        end_sym = get_end_symbol(string,type)

        if end_sym:
            num_order = end_sym_analysis.get(end_sym, None)
            if num_order is None:
                end_sym_analysis[end_sym] = (1, i)
            else:
                end_sym_analysis[end_sym] = (num_order[0] + 1, i)
        else:
            candidate_id_no_tail_sym.append(i)

    # print(end_sym_analysis)

    prior_replace_item = ''
    expect_sym_pair_key = ''
    for k in end_sym_analysis.keys():
        '''
        配对字符优先级别最高，以替换方式进行
        '''
        sym_may2expect_key = is_contain_pair_sym(k)
        if sym_may2expect_key is not None:
            if len(k) > len(prior_replace_item):
                prior_replace_item = k
                expect_sym_pair_key = sym_may2expect_key

    # 优先替换项
    if prior_replace_item:
        expect_sym_pair_value = exp_replace_dict[expect_sym_pair_key]
        expect_sym_pair_value = expect_sym_pair_value[0]+'  '+expect_sym_pair_value[1]
        prior_replace_item = prior_replace_item.replace(expect_sym_pair_key,expect_sym_pair_value)
        return prior_replace_item,None
    else:
        end_sym_analysis = sorted(end_sym_analysis.items(), key=lambda x: (x[1][0], -x[1][1]), reverse=True)

        symbol2add = end_sym_analysis[0][0] if len(end_sym_analysis)>0 else ''

        return symbol2add,candidate_id_no_tail_sym

def deRepeat(sym):
    ret = ''
    for c in sym:
        if c in equal_sym_dict:
            ret += equal_sym_dict[c]
        else:
            ret += c

    retInList = list(set(ret))
    retInList.sort(key=ret.index)

    return ''.join(retInList)

def replace_candidate_with_sym(candidates,sym,type='tail'):
    replace_sym_candidates = []
    # 去除重复标点,特别是半圆括号
    sym = deRepeat(sym)

    for candidate in candidates:
        de_sym_candidate = de_end_symbol(candidate, type)
        if type=='tail':
            replace_sym_candidates.append(de_sym_candidate+sym)
        else:
            replace_sym_candidates.append(sym+de_sym_candidate)

    return replace_sym_candidates

def add_sym2candidate(candidate=None):
    if (candidate is None) or (not len(candidate)):
        return candidate

    sym2add, to_add_candidate_id = find_sym2add(candidate,type='tail')
    # to_add_candidate_id is None, 替换为成对标点
    if to_add_candidate_id is None:
        return replace_candidate_with_sym(candidate,sym2add,type='tail')
    else:
        if sym2add:
            for i in to_add_candidate_id:
                candidate[i] =candidate[i].strip(' ')+sym2add

    sym2add, to_add_candidate_id = find_sym2add(candidate, type='head')

    if to_add_candidate_id is None:
        return replace_candidate_with_sym(candidate,sym2add,type='head')
    else:
        if sym2add:
            for i in to_add_candidate_id:
                candidate[i] = sym2add + candidate[i].strip(' ')

    return candidate


if __name__ == '__main__':

    test_example_list = [['performance', 'performance.', 'performance,', 'performancec', 'perfornmance'],
                    ['全球共分几大板块 （','全球共分几大板块（）','全球共分几大板块）','全球共分几大板块','全球共分几大板块（）.'],
                    ['全球共分几大板块 （', '全球共分几大板块。（）', '全球共分几大板块）', '全球共分几大板块', '全球共分几大板块（）.'],
                    ['(1)粒子进入？','1)粒子进入','（1）粒子进入?','(1)粒子进入'],
                    ['全球共分几大板块 （', '全球共分几大板块（）.', '全球共分几大板块）.', '全球共分几大板块', '全球共分几大板块(.)'],
                    [')粒子进入','（粒子进入','（粒子进入','）粒子进入','( )粒子进入'],
                    [')粒子进入', '（粒子进入', '（粒子进入', '）粒子进入', '(）粒子进入'],
                    [')粒子进入', '（粒子进入', '（粒子进入', '）粒子进入', '(> 粒子进入'],
                    ['序是（  ）', '序是（   ）', '序是（  ））', '序是（    ）', '序是（  ） ）'],
                    ['）', '﹚', '〕', ')', ''],
                    ['[', '[】', '[I', '[〕', '[J'],
                    ['[', '[〕', '[】', '[I', '[J'],
                    ['C. A manager ', 'C. A manager', 'C. A manager.', 'C.A manager ', 'C. A manager. '],
                    ['﹚', '）', '', '〕', ' ﹚'],
                    ]
    for test_example in test_example_list[-1:]:
        print(add_sym2candidate(test_example))
    print(equal_sym_dict)
    # print(exp_replace_dict)