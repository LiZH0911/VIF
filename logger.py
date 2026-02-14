import logging
import time
import os

def setup_logging(log_dir='./'):
    log_filename = 'program_{}.log'.format(time.strftime('%Y%m%d_%H%M%S'))
    log_path = os.path.join(log_dir, log_filename)

    # 日志基础配置
    logging.basicConfig(level=logging.DEBUG, # 日志级别，DEBUG为10，往上为INFO、WARNING、ERROR、CRITICAL
                        format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s(%(lineno)d): %(message)s', # 输出格式
                        filename=log_path, # 日志文件路径
                        filemode='a') # 文件打开模式，'a'追加，'w'覆盖，默认追加
    logging.root.addHandler(logging.StreamHandler()) # 同时输出到日志和终端

# 示例
if __name__ == '__main__':
    setup_logging(log_dir='./')
    logger = logging.getLogger()  # 创建日志记录器对象，默认对象名称为root
    print(f"日志对象名称: {logger.name}")
    print(f"级别: {logger.level}")
    logger.info('Started')
    logger.info('Finished')






