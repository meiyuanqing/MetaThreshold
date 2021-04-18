#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/4/18
Time: 22:02
File: toDocFormat.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

将元分析结果整理到WORD文档中去，按如下顺序：
    31个规模度量(表A1)、18个复杂性度量(表A2)、18个耦合性度量(表A3)、19个继承性度量(表A4)和两个内聚性度量(表A5)，一共88个度量。

"""
import time


def doc_format(dir_file):
    import os
    import csv
    import xlwt
    import numpy as np
    import pandas as pd

    # read_csv(path, keep_default_na=False, na_values=[""])  只有一个空字段将被识别为NaN
    df = pd.read_csv(dir_file, keep_default_na=False, na_values=[""])
    file = dir_file.split("\\")[-1]
    directory = dir_file.replace(file, "")
    os.chdir(directory)
    print(os.getcwd())
    print(directory)
    print(file)
    columns = df.columns
    print(columns)

    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet_size = workbook.add_sheet("size", cell_overwrite_ok=True)
    size = ["AvgLine", "AvgLineBlank", "AvgLineComment", "AvgSLOC", "CountClassBase", "CountDeclClassVariable",
            "CountDeclInstanceVariable", "CountDeclMethodDefault", "CountDeclMethodPrivate", "CountDeclMethodProtected",
            "CountLine", "CountLineBlank", "CountLineCodeDecl", "CountLineCodeExe", "CountLineComment", "CountSemicolon",
            "CountStmtDecl", "CountStmtExe", "NA", "NAIMP", "NCM", "NIM", "NLM", "NM", "NMIMP", "NMNpub", "Nmpub", "NTM",
            "NumPara", "SLOC", "stmts"]
    for i in range(size):
        booksheet_size.write(df[df["metric"] == size])
    workbook.save(directory + "doc_" + file.replace("csv", "xls"))

if __name__ == '__main__':
    s_time = time.time()
    m_dir_file = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\ValeThresholdMinified\\Vale_metaThresholds.csv"
    doc_format(m_dir_file)
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of toDocFormat.py!\n",
          "The execution time of toDocFormat.py script is ", execution_time)
