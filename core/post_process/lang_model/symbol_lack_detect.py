pair_symbol = ['（','）','(',')','【','】','[',']']

def fill_tail_lack(string):
    '''
    已经完成中英文替换基础上，
    还没遇到非标点或者空格前，（目前只考虑圆括号丢失情况，较普遍）
    从尾到头进行配对搜索,不完整配对考虑填补
    :param string:
    :return:
    '''
    counter =[0] * len(pair_symbol)
    stop_id = len(string)
    reversed_string = string[::-1]
    meet_space = False
    for id,char in enumerate(reversed_string):
        if char!=' ' and (char not in pair_symbol):
            stop_id =id
            break
        else:
            if char!=' ':
                counter[pair_symbol.index(char)]+=1
            elif char==' ':
                right_sym_num =0
                for i in range(0,len(pair_symbol),2):
                    right_sym_num+= counter[i]
                if right_sym_num:
                    meet_space = True

    # 非结尾符号部分
    nosymbol_part = reversed_string[stop_id:]
    symbol_part = reversed_string[:stop_id]

    ret_string =symbol_part + nosymbol_part

    for i in range(0,len(pair_symbol),2):
        if counter[i]!=counter[i+1]:
            replace_item = pair_symbol[i+1] + '  ' + pair_symbol[i]
            if counter[i]<counter[i+1]:
                if meet_space:
                    symbol_part = symbol_part.replace(pair_symbol[i+1],replace_item)
            else:
                symbol_part =symbol_part.replace(pair_symbol[i], replace_item)
            ret_string = symbol_part + nosymbol_part
            break
        else:
            continue

    return ret_string[::-1]

def fill_tail_lackV2(string):
    '''
    已经完成中英文替换基础上，
    还没遇到非标点或者空格前，（目前只考虑圆括号丢失情况，较普遍）
    从尾到头进行配对搜索,不完整配对考虑填补
    单个右侧符号必须与向左第一的文字符号间存在2个（默认）空格参会补齐
    :param string:
    :return:
    '''
    counter =[0] * len(pair_symbol)
    stop_id = len(string)
    reversed_string = string[::-1]
    # meet_space = False
    left_space_num = 0

    for id,char in enumerate(reversed_string):
        if char!=' ' and (char not in pair_symbol):
            stop_id =id
            break
        else:
            if char!=' ':
                counter[pair_symbol.index(char)]+=1
            elif char==' ':
                right_sym_num =0
                for i in range(0,len(pair_symbol),2):
                    right_sym_num+= counter[i+1]

                if right_sym_num:
                    left_space_num +=1
                    # meet_space = True

    # 非结尾符号部分
    nosymbol_part = reversed_string[stop_id:]
    symbol_part = reversed_string[:stop_id]

    ret_string =symbol_part + nosymbol_part

    for i in range(0,len(pair_symbol),2):
        if counter[i]!=counter[i+1]:
            replace_item = pair_symbol[i+1] + '  ' + pair_symbol[i]
            if counter[i]<counter[i+1]:
                if left_space_num>1:
                    symbol_part = symbol_part.replace(pair_symbol[i+1],replace_item)
            else:
                symbol_part =symbol_part.replace(pair_symbol[i], replace_item)
            ret_string = symbol_part + nosymbol_part
            break
        else:
            continue

    return ret_string[::-1]

if __name__ == '__main__':
    test_examples = ['test 1 （ ）',
                     'test 2 ( ) ',
                     'test 3 （',
                     'test 4 ）',
                     'test 5 (  ',
                     'test 6  ) ',
                     'test 7 【  ',
                     'test 8 ]  ',
                     'test 9 (  ',
                     'test 10(  ',
                     'test 11)  ',
                     'test 12  )  ',
                     ]

    for example in test_examples:
        # string = fill_tail_lack(example)
        string = fill_tail_lackV2(example)
        print(string)