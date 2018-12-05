import codecs
# 把文档中的数据去重
def deal_datas(name):
    datas = codecs.open('%s.txt' %(name), "r", encoding='utf-8')
    clear_datas=set(data.strip() for data in datas)
    print(clear_datas)
    clear_datas
    output_datas=codecs.open('%s1.txt' %(name),'w',encoding='utf-8')
    for clear_data in clear_datas:
        output_datas.write(clear_data+'\n')
if __name__=='__main__':
    deal_datas('母婴')
    deal_datas('数码')
