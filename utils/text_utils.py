"""
从文字识别结果中提取公式
"""
import os
import glob
#import tqdm

formula_set = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789<>&+-βαφγνΩλπµ.')
latex_dict = {'β': '\\beta', 'α': '\\alpha', 'φ': '\\phi', 'γ': '\\gamma', 'ν': '\\nu', 'Ω': '\\Omega', 'λ': '\\lambda', 'π': '\\pi', 'µ': '\\mu'}

def get_tihao_set_1():
    tihao_set = []
    for i in range(100):
        tihao_set.append(str(i))
        tihao_set.append(str(i) + '.')
        tihao_set.append(str(i) + '、')
    options = ['A', 'A.', 'A、', 'B', 'B.', 'B、', 'C', 'C.', 'C、', 'D', 'D.', 'D、', 'a', 'a.', 'a、', 'b', 'b.', 'b、',
               'b', 'b.', 'b、', 'c', 'c.', 'c、', 'd', 'd.', 'd、']
    tihao_set.extend(options)
    return set(tihao_set)

def get_tihao_set_2():
    tihao_set = []
    for i in range(50):
        tihao_set.append('(' + str(i) + ')')
        tihao_set.append('(' + str(i) + ').')
        tihao_set.append('(' + str(i) + ')、')
        tihao_set.append('（' + str(i) + '）')
        tihao_set.append('（' + str(i) + '）.')
        tihao_set.append('（' + str(i) + '）、')
        tihao_set.append('(' + str(i) + '）')
        tihao_set.append('(' + str(i) + '）.')
        tihao_set.append('(' + str(i) + '）、')
        tihao_set.append('（' + str(i) + ')')
        tihao_set.append('（' + str(i) + ').')
        tihao_set.append('（' + str(i) + ')、')
        tihao_set.append(str(i) + ')')
        tihao_set.append(str(i) + ').')
        tihao_set.append(str(i) + ')、')
        tihao_set.append(str(i) + '）')
        tihao_set.append(str(i) + '）.')
        tihao_set.append(str(i) + '）、')
        tihao_set.append(')' + str(i))
        tihao_set.append(').' + str(i))
        tihao_set.append(')、' + str(i))
        tihao_set.append('）' + str(i))
        tihao_set.append('）.' + str(i))
        tihao_set.append('）、' + str(i))
    tihao_set.extend(['(A)', '(B)', '(C)', '(D)', '(A）', '(B）', '(C）', '(D）', '（A)', '（B)', '（C)', '（D)', '（A）',
                      '（B）', '（C）', '（D）'])
    return set(tihao_set)

tihao_set_1 = get_tihao_set_1()  # 大题号
tihao_set_2 = get_tihao_set_2()  # 小题号及选项号

def textExtrFormula(text, tihao_set_1=tihao_set_1, tihao_set_2=tihao_set_2, subject='数学'):
    """
    :param text: 需要提取公式后处理的文字行
    :param subject: 科目信息
    :return:
    """
    if subject == "生物":
        return text
    copy_text = text
    text_str = ''
    formula_str = ''
    start_str = ''
    # 先把开头的题号提取出来
    for i in range(5):
        temp_str = text[:(3-i+1)]
        if temp_str in tihao_set_2:
            start_str = temp_str
            break
    text = text[len(start_str):]
    new_text = start_str
    start_bool = False
    for i in range(len(text)):
        if text[i] in formula_set:
            if text[i] in latex_dict:
                formula_str = formula_str + latex_dict[text[i]] + ' '
            else:
                formula_str = formula_str + text[i]
            if i > 0:
                if text[i-1] not in formula_set:
                    new_text = new_text + text_str
                    text_str = ''
            if i == 0:
                start_bool = True
        else:
            text_str = text_str + text[i]
            if i > 0:
                if text[i-1] in formula_set:
                    new_text, start_bool = update_fromula(new_text, formula_str, start_bool, tihao_set_1, subject, i, copy_text, start_str)
                    formula_str = ''
    if len(text_str) > 0:
        new_text = new_text + text_str
    if len(formula_str) > 0:
        new_text, start_bool = update_fromula(new_text, formula_str, start_bool, tihao_set_1, subject, i, copy_text, start_str)
    return new_text

def update_fromula(new_text, formula_str, startbool, tihao_set_1, subject, index, text, start_str):
    """
    :param new_text: 前面已经公式化过的字符串
    :param formula_str: 当前单纯的数字或字母的字符串
    :param startbool: formula_str是否位于文字行起始位置
    :param tihao_set_1: 需要排除的题号字符串集合
    :param subject: 科目信息
    :param i: formula_str最后一个字符在文字行的位置
    :param text: 文字行
    :param start_str: 判断前面是否已经提取出题号了，这里就不需要再去除题号
    :return:
    """
    if startbool:#起始位置的题号排除
        if formula_str in tihao_set_1:
            new_text = new_text + formula_str
        else:
            if len(start_str)==0:
                split = formula_str.split('.', 1)
                if len(split) > 1:  # 处理类似 10.A的情况
                    if split[0] in tihao_set_1:
                        new_text = new_text + split[0] + '.'
                        formula_str = split[1]
            if subject == '化学':
                formula_str = '\\rm ' + formula_str
            new_text = new_text + '${' + formula_str + '}$'
        startbool = False
    else:
        option_set_1 = {'B': ['A.', 'A．', 'a.', 'a．'], 'C': ['B.', 'B．', 'b.', 'b．'], 'D': ['C.', 'C．', 'c.', 'c．']}
        option_set_2 = {'B.': ['A.', 'A．', 'a.', 'a．'], 'C.': ['B.', 'B．', 'b.', 'b．'], 'D.': ['C.', 'C．', 'c.', 'c．']}
        option_bool = False
        if formula_str in option_set_1:#排除中间的选择题选项号
            for i in range(4):
                if option_set_1[formula_str][i] in text[:index]:
                    option_bool = True
        if formula_str in option_set_2:#排除中间的选择题选项号
            for i in range(4):
                if option_set_2[formula_str][i] in text[:index]:
                    option_bool = True
        if option_bool:
            new_text = new_text + formula_str
        else:
            if subject == '化学':
                formula_str = '\\rm ' + formula_str
            new_text = new_text + '${' + formula_str + '}$'
    return new_text, startbool


if __name__ == '__main__':
    pass
    # datadir = r'E:\formula\后处理测试\high_math_res'
    # savedir = r"E:\formula\后处理测试\high_math"
    # txtfiles = glob.glob(os.path.join(datadir, '*.txt'))
    #
    # for txtfile in tqdm.tqdm(txtfiles):
    #     basename = os.path.basename(txtfile)
    #     basename = basename.replace('_gz_text', '')
    #     sf = open(os.path.join(savedir, basename), 'w', encoding='utf-8')
    #     lines = open(txtfile, 'r', encoding='utf-8').read().split('\n')
    #     for line in lines:
    #         line_split = line.split(', ', 4)
    #         if len(line_split) <= 4:
    #             sf.write(line + '\n')
    #             continue
    #         if line_split[4].startswith('${'):
    #             sf.write(line + '\n')
    #             continue
    #         new_label = textExtrFormula(line_split[4].strip(' '), tihao_set_1, tihao_set_2)
    #         sf.write(str(line_split[0]) + ', ' + str(line_split[1]) + ', ' + str(line_split[2]) + ', '
    #                  + str(line_split[3]) + ', ' + new_label + '\n')
    #     sf.close()
    # newlabel = textExtrFormula('（A）10', tihao_set_1, tihao_set_2)
    # print(newlabel)