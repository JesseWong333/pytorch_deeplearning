symbol_list = "',.;:\\|/?\"`^&*()<>{}[]!@#$%*-_+=，。；：“”‘’｛｝【】、《》？！￥……（）——~·"
def desymbol(string):
    ret = ''
    for char in string:
        if char not in symbol_list:
            ret+=char
    return ret

if __name__ == '__main__':
    import jieba
    test_string_list = ['To be more __67 _( efficiency ) ','C总分150分，150分钟完卷）', '（总分150分，150分钟完卷）']
    for string in test_string_list:
        ret = desymbol(string)
        print(string,ret,len(list(jieba.cut(string,HMM=False))))