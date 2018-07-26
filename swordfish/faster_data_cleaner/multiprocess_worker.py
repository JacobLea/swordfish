# -*- encoding: utf-8 -*-

import subprocess
import multiprocessing
import time
import os
import sys
from sys import argv


class Jobs(object):
    data_name = ""
    out_put_file = ""
    data_source = []
    data_dispatcher = lambda data, job_id: data
    data_maintenance = lambda data, path: None
    data_loader = lambda path: {}
    data_cleaner = lambda data: data
    data_stacker = lambda data_list: data_list


def _silent_remove_dir(d):
    if isinstance(d, str):
        if os.path.exists(d):
            if os.path.isfile(d):
                os.remove(d)
                return
            for f in os.listdir(d):
                _silent_remove_dir('/'.join([d, f]))
            os.removedirs(d)
    elif isinstance(d, list):
        [_silent_remove_dir(f) for f in d]


class _Constants(object):
    is_main = None
    jobs_pool = None
    cpu_num = multiprocessing.cpu_count()
    current_job_id = None
    prj_path = None
    TEMP_DIR = "./temp"
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    IS_WINDOWS = sys.platform.lower().startswith('win')
    MAIN_PY_FILE = os.path.split(sys._getframe().f_code.co_filename)[1]


def _iterate_data(data):
    for chunk in data:
        # 根据进程id筛选数据
        if _Constants.cpu_num > 1:
            # data dispatcher
            rtn = Jobs.data_dispatcher(chunk, _Constants.current_job_id)
        else:
            rtn = chunk
        return rtn


def _get_file_name(idx):
    return os.path.join(_Constants.prj_path, f'{_Constants.TEMP_DIR}/{Jobs.data_name}_{idx}.temp')


# 定义子数据导出任务
def _sub_process():
    data = Jobs.data_stacker([Jobs.data_cleaner(_iterate_data(Jobs.data_source))])
    if _Constants.is_main:
        return data
    else:
        Jobs.data_maintenance(data)


# 创建一个线程任务运行当前py文件
def _extract_data(i):
    py_file = os.path.join(_Constants.prj_path, _Constants.MAIN_PY_FILE)
    cmd = f'python {repr(py_file)} {i}'
    # return subprocess.Popen(cmd, shell=~Constants.IS_OFFLINE)
    return subprocess.Popen(cmd, shell=~_Constants.IS_WINDOWS, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# 创建其他线程任务
def _batch_extract_data():
    _Constants.jobs_pool = []
    if _Constants.cpu_num > 1:
        for i in range(_Constants.cpu_num - 1):
            _Constants.jobs_pool.append(_extract_data(i))


# 加载其他进程数据
def _loads_other_data():
    data_list = []
    done_jobs = list()
    while len(done_jobs) != len(_Constants.jobs_pool):
        for i, job in enumerate(_Constants.jobs_pool):
            if (i not in done_jobs) & (job.poll() is not None):
                data_list.append(Jobs.data_loader(_get_file_name(i)))
                done_jobs.append(i)
        time.sleep(0.05)
    _Constants.jobs_pool.clear()
    return data_list


# 主进程入口
def _main_process():
    # 开启其他CPU任务
    _batch_extract_data()
    # 执行主进程任务
    main_data = _sub_process()
    # 获取其他进程结果
    other_data = _loads_other_data()
    _silent_remove_dir(_Constants.TEMP_DIR)
    Jobs.data_maintenance(Jobs.data_stacker(other_data + [main_data]), Jobs.out_put_file)


# 判断当前进程性质
def _test_is_main():
    for i, s in enumerate(reversed(argv)):
        if s.isnumeric():
            _Constants.prj_path = os.path.sep.join(argv[-i - 2].split(os.path.sep)[:-1])
            _Constants.current_job_id = int(s)
            _Constants.is_main = False
            return False
        elif s.endswith('.py'):
            _Constants.prj_path = os.path.abspath('.')
            _Constants.current_job_id = _Constants.cpu_num - 1
            _Constants.is_main = True
            return True


def main():
    if _test_is_main():
        _main_process()
    else:
        _sub_process()


if __name__ == '__main__':
    main()
