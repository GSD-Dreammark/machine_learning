import requests
import re
import cache
import request
#  不能添加空格
# def cacheip(timeout=60*60*24, key='ip%stype%s'):
#     def out_func(func):
#         def in_func(*args, **kwargs):
#             #获取请求者ip
#             ip = request.remote_addr
#             typename = request.form.get('typename')
#             #设置key
#             cache_key = key%(ip,typename)
#             #首次请求get获取不到数据
#             value = cache.get(cache_key)
#             #程序出错返回null
#             res = func(*args, **kwargs)
#             #发送短信的条件程序出错并且key首次出现
#             if value is None and res is 'null':
#                 ip = request.remote_addr
#                 typename = request.form.get('typename')
#                 data = request.form.get('data')
#                 #将set key值确保同一个ip多次请求只发一次短信,每天重置过期时间
#                 cache.set(cache_key, ip, timeout=timeout)
#                 #发短信
#                 send_move(ip,typename,data)
#             return res
#         return in_func
#     return out_func
def send_move(ip,typename,data):
    url = 'http://api.yuanrongbank.com/sms/sendMessage'
    params = {
        #手机号
        'iphoneNum': 18701344187,
        'token': 'yx-admin-mall',
        #用户名
        'user': '王星月',
        #短信内容
        'content': "亲，起床了！你的程序出错了，出错ip：{0}，程序名字：{1}，错误信息：{2}".format(ip,typename,data),
        'source':'短信报警'
    }
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"} #设置请求头
    requests.post(url=url,params=params,headers=headers)
    print('ok')
def errorwarning(func):
    def in_func(*args, **kwargs):
        #获取被装饰函数的返回值
        res = func(*args, **kwargs)
        #如果返回值是出错结果则发送短信
        if len(res)>=0:
            print("发送短信")
            send_move('118','爬虫ttkp__spider',res)
        return res
    return in_func

@errorwarning
def text():
    try:
        2/0
    except Exception as e:
        error=re.sub(' ',',',str(e))
        return error
if __name__=="__main__":
    text()