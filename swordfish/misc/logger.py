# -*- coding: utf-8 -*-

import logging
import os, sys, time
from swordfish.misc.base_class import Singleton


class Logger(Singleton):
    _logger_pool = {}

    def get_logger(self, name: str, name_space='root', base_dir="../logs/", level=logging.DEBUG):
        name = name.replace(".log", "")
        date = time.strftime('%Y-%m-%d', time.localtime())
        log_key = '.'.join([name_space, name, date])
        if log_key in self._logger_pool.keys():
            if self._logger_pool[log_key].level != level:
                _log = logging.getLogger(__name__)
                _log.warning(f'[Warning]: The current logging level is {self._logger_pool[log_key].level},' +
                             f' which is different with the new one {level}. No change would happen.')
                del _log
            return self._logger_pool[log_key]
        else:
            logger = logging.getLogger(log_key)
            logger.setLevel(level if level else logging.DEBUG)
            if not logger.handlers:
                path = os.path.join(base_dir, '.'.join([name_space, name, date, 'log']))
                stout_handler = logging.StreamHandler(sys.stdout)
                file_handler = logging.FileHandler(path)
                fmt = logging.Formatter("%(asctime)s, [%(levelname)s], @%(funcName)s: %(message)s")
                for handler in (stout_handler, file_handler):
                    handler.setFormatter(fmt)
                    handler.setLevel(logging.DEBUG)
                    logger.addHandler(handler)
            self._logger_pool[log_key] = logger
            return logger
