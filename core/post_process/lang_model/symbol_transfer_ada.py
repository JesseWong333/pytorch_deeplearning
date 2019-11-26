cn2en_symbol = {
    '，': ',', '：': ':', '；': ';',
    '‘': '\'', '’': '\'', '”': '"', '“': '"',
    '（': '(', '）': ')', '｛': '{', '｝': '}',
    '？': '?', '！': '!',
}
# 不用于转换，只用于判断前一个符号是cn/en类型
en_spec_symbol_list = '.．$'
# 不用于影响之前符号类型判断的标点列表
symbol_not_effect2_prev_list = '_'

numbers = [str(i) for i in range(10)]
en2cn_symbol = dict((item[1], item[0]) for item in cn2en_symbol.items())
alphabet = 'abcdefghijklmnopqrstuvwxyz' \
           'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def get_latter_type(string, prev_type, cur_step, latter_step=5):
    meet_en = None
    for latter in string[cur_step + 1:cur_step + 1 + latter_step]:
        if latter == ' ' or latter in numbers or latter in en2cn_symbol or latter in cn2en_symbol:
            continue
        if latter in alphabet or latter in en2cn_symbol:
            meet_en = True
            break
        else:
            meet_en = False
            break
    if meet_en is None:
        return prev_type
    elif meet_en:
        return 'en'
    else:
        return 'cn'


def is_string_has_num(string, cur_id, prev_step, latter_step):
    left = False
    right = False

    # print(string[max(0, cur_id - 1 - prev_step):cur_id],string[cur_id + 1:cur_id + 1 + latter_step])
    for c in string[max(0, cur_id - 1 - prev_step):cur_id]:
        if c == ' ':
            continue
        if c in numbers:
            left = True
    if not left:
        return False

    for c in string[cur_id + 1:cur_id + 1 + latter_step]:
        if c == ' ':
            continue
        if c in numbers:
            right = True
    if not right:
        return False

    return True


def replace_symbol_ada(example_string, subject=None):
    '''
    根据标点前一字符类型进行替换操作
    :param example_string: 输入
    :return: string : 替换后的字符串
    '''
    string = ''
    prev_char_type = 'en' if subject == '英语' else ''

    for id, char in enumerate(example_string):
        # for debug
        # print(char,prev_char_type,char in en2cn_symbol,char in cn2en_symbol)

        latter_type = get_latter_type(example_string, prev_char_type, id, latter_step=5)
        is_num_before_after = is_string_has_num(example_string, id, prev_step=2, latter_step=2)

        if id > 0 and char == '。' and is_num_before_after:
            # 特殊情况 如 1。2
            string += '.'
        elif id > 0 and char == '.' and is_num_before_after:
            # 1.2 排除前后有中文
            # print(char,is_num_before_after)
            string += char
        elif prev_char_type == 'cn' and char in en2cn_symbol and latter_type == 'cn':
            char = en2cn_symbol[char]
            string += char
        elif prev_char_type == 'en' and char in cn2en_symbol and latter_type == 'en':
            char = cn2en_symbol[char]
            string += char
        else:
            string += char

        # print(char)

        # 记录标点前一字符是中英何种类型
        if (char in alphabet) or (char in en2cn_symbol) or (char in en_spec_symbol_list):
            prev_char_type = 'en'
        elif char == ' ' or (char in numbers) or (char in symbol_not_effect2_prev_list):
            pass
        else:
            prev_char_type = 'cn'

    # 特殊情况
    if subject == '英语':
        string = string.replace('。。', '..')
    string = string.replace('—-', '——')
    return string


if __name__ == '__main__':
    test_example = ['what the bi,',
                    'what the bi，',
                    'what the bi ，',
                    'what the bi ,',
                    '什么鬼，',
                    '作甚,',
                    '丢什么 ;',
                    '丢什么 ；',
                    'hey, 顶, siri，',
                    'en,，',
                    ' Switzerland, April 18， 2017。 The ',
                    'The shorter boy got up and took a_ 45 _。He fell'
                    '(多方式的)。She',
                    '1。5',
                    '1。 5',
                    ' 第二节 （共10小题；每小题1。 5分，满分15分',
                    '16. 有的人是单眼皮，有',
                    'C.$ 13.',
                    'D．.鹦鹉学舌',
                    '“Use the good manncrs that your mother _ 70 _  ( teach） you."',
                    '“Use the good manncrs that your mother _ 70 _  ( teach） you.”',
                    '“Use the good manncrs that your mother _ 70__  ( teach） you."',
                    '“Use the good manncrs that your mother _ 70_  ( teach） you."',
                    '“Use the good manncrs that your mother _ 70__  ( teach） you.”',
                    ]
    for exmaple in test_example[-2:-1]:
        string = replace_symbol_ada(exmaple, subject='英语')
        print('英语科目:{}'.format(string))
        string = replace_symbol_ada(exmaple)
        print('非英语科目:{}'.format(string))
