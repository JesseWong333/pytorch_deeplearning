import wordninja

# ' 除外，wordninja不会对其进行忽略
symbol_list = ",.;:\\|/?\"`^&*()<>{}[]!@#$%*-_+=，。；：“”‘’｛｝【】、《》？！￥……（）——~·"
cn2en_symbol = {
    '，':',','。':'.','：':':','；':';',
    '‘':'\'','’':'\'','”':'"','“':'"',
    '（':'(','）':')','《':'<','》':'>','｛':'{','｝':'}','【':'[','】':']',
    '？':'?','！':'!',
}
en2cn_symbol =dict((item[1],item[0]) for item in cn2en_symbol.items())
# 英文起始位:[11:]
# 数字起始位:[1:11]
alphabet= '\'0123456789' \
          'abcdefghijklmnopqrstuvwxyz' \
          'ABCDEFGHIJKLMNOPQRSTUVWXYZ' \


def split(string):
    orign_len =len(string)

    if orign_len<=1:
        return string

    split_ret =' '.join(wordninja.split(string))

    if len(split_ret)<1:
        return string

    after_add_dict = ['' ]*orign_len

    for id,c in enumerate(string):
        if c not in alphabet:
            after_add_dict[id]=c

    string_id=split_string_id = 0
    tmp_string = ''
    while string_id < len(string) and split_string_id<len(split_ret):
        # print(string[string_id],split_ret[split_string_id] )
        if string[string_id]==split_ret[split_string_id] :

            tmp_string+=string[string_id]
            string_id+=1
            split_string_id+=1

        elif split_ret[split_string_id]!=' ' and string[string_id]!=split_ret[split_string_id]:
            tmp_string += string[string_id]
            string_id += 1

        elif split_ret[split_string_id]==' ':
            # 标点符号先回补
            if string[string_id] in symbol_list:
                tmp_string+=string[string_id]
                string_id+=1
            # 空格后是字母时才会添加
            if string_id < len(string) and string[string_id] in alphabet[-62:]:
                tmp_string +=' '
            split_string_id +=1

    if string_id<len(string) and split_string_id>=len(split_ret):
        tmp_string +=string[string_id:]


    ret= ''
    for c in tmp_string:
        if c in cn2en_symbol:
            ret += cn2en_symbol[c]
        else:
            ret+=c

    # print('分词结果: {}'.format(split_ret))
    # print('原句: {}'.format(string))
    # print('分词回补标点结果:{}'.format(tmp_string))
    # print('替换中文标点:{}'.format(ret))
    return ret
if __name__ == '__main__':
    test_string = '哈 罗whatthehell1行街shit；'
    split_ret = split(test_string)
    print(test_string)
    print(split_ret)
