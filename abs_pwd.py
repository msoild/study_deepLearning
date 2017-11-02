# encoding=utf-8
import os


def get_abs_pwd(dirname):
    '''
    :param dirname:
    :return:
    '''
    return os.path.dirname(os.getcwd())+'/'+dirname
