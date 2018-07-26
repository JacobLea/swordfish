# -*- coding: utf-8 -*-


class Singleton(object):
    _instances = {}

    def __new__(cls_, *args, **kwargs):
        if cls_ not in cls_._instances:
            cls_._instances[cls_] = super(Singleton, cls_).__new__(cls_, *args, **kwargs)
        return cls_._instances[cls_]
