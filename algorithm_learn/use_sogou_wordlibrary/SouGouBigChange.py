#coding=utf-8
# python2
import  os
import  SougouChangeTxt

def  main():
    #获取文件路径
    filepath=os.getcwd()+'\\Film'
    #获取当前的文件路径的文件名称，返回为list
    num=0
    car_word = []
    filenames_word=os.listdir(filepath)
    for filepath_word in filenames_word:#每个类别的下的
        filepath_end=filepath+'\\'+filepath_word
        GTable=SougouChangeTxt.main(filepath_end)
        for word in GTable:
            num+=1
            car_word.append(word[2])
    f = open(os.getcwd()+'/Film/{0}.txt'.format('11'), 'w')
    print  len(set(car_word))
    for line in set(car_word):
        f.write(line)
        f.write('\n')
    f.close()
    print num

if __name__=='__main__':
    main()




