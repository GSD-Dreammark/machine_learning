# 测试 python3的input降级后输入字符类型需要加引号
ss=input('输入年龄：')
print(type(ss))
# python3降级使用Python2的raw_input
ss=eval(input('输入年龄：'))
print(type(ss))