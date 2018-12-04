import logging
import sys
import time
import os
import logging.handlers


def get_logger(name):
    # 创建一个logger
    logger = logging.getLogger(name)
    # log的总等级开关
    logger.setLevel(logging.DEBUG)
    # 定义handler的输出格式
    formatter = logging.Formatter(
        '[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')

    # 创建一个handler，用于输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # 创建一个handler，用于写入日志文件
    # file_handler = logging.FileHandler(path)
    # file_handler.setLevel(logging.WARNING)
    # file_handler.setFormatter(formatter)
    # rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    rq=name
    log_name =  rq + '.log'
    time_handler = logging.handlers.TimedRotatingFileHandler(log_name, when='D', interval=1, backupCount=15, utc=False)
    time_handler.setLevel(logging.ERROR)
    time_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(stream_handler)
    logger.addHandler(time_handler)
    # logger.addHandler(file_handler)
    return logger

# logger = get_logger('doublewei', 'log/shortvideo.log')

# logger = get_logger('shortvideo', '../log/shortvideo.log')

