#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
__title__ = 'run'
__author__ = 'JieYuan'
__mtime__ = '2019/4/17'
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def main(n_iter):
    data = make_classification(10 ** 6, 10 ** 3)
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=n_iter)
    clf.fit(*data)
