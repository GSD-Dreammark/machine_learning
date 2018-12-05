#coding=utf-8
# python2
import  os
import  SougouChangeTxt

def  main():
    #获取文件路径
    filepath=os.getcwd()+u'\\WordBank'
    #获取当前的文件路径的文件名称，返回为list
    filenames=os.listdir(filepath)
    num=0
    for filename in filenames:#类别
        car_word = []
        filename_path=filepath+'\\'+filename
        filenames_word1=os.listdir(filename_path)
        for filepath_word1 in filenames_word1:
            filename_path1=filename_path+'\\'+filepath_word1
            filenames_word = os.listdir(filename_path1)
            for filepath_word in filenames_word:#每个类别的下的
                filepath_end=filename_path1+'\\'+filepath_word
                if os.path.isdir(filepath_end):
                    filenames_word2 = os.listdir(filepath_end)
                    for filepath_word2 in filenames_word2:  # 每个类别的下的/
                        filepath_end1 = filepath_end + '\\' + filepath_word2
                        GTable = SougouChangeTxt.main(filepath_end1)
                        for word in GTable:
                            num += 1
                            car_word.append(word[2])
                else:
                    GTable = SougouChangeTxt.main(filepath_end)
                    for word in GTable:
                        num += 1
                        car_word.append(word[2])
    f = open(os.getcwd()+'\\{0}.txt'.format('all'), 'a')
    print  len(set(car_word))
    for line in set(car_word):
        f.write(line)
        f.write('\n')
    f.close()
    print num
if __name__=='__main__':
    main()




