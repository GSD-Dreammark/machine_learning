# 这段代码很奇怪
# 枚举类型 是一个对象，没有返回值
# 一般用于读取文件计数
la=enumerate(['ss','dd','ee'])
# la 是地址
print(la)
print(list(la))
# la 被重置了
print(la)
for i,line in la:
    print(i)
    print(line)
print(list(la))
