#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/3/17
Time: 10:43
File: data_preprocessing.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Sort out the executable metric set
"""

import os
import csv
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

working_dir = "F:\\NJU\\MTmeta\\experiments\\testing\\"
# working_dir = "F:\\NJU\\MTmeta\\experiments\\training_data\\"
result_dir = "F:\\NJU\\MTmeta\\experiments\\testings\\"
# result_dir = "F:\\NJU\\MTmeta\\experiments\\trainings\\"

os.chdir(working_dir)
print(os.getcwd())


def is_equal(m, n):
    if m == n:
        return 1
    else:
        return 0

with open(working_dir + "List.txt") as l:
    lines = l.readlines()

for line in lines:

    file = line.replace("\n", "")
    print(repr(line))
    print(file)

    with open(working_dir + "uperl_und_liu_" + file, 'r', encoding="ISO-8859-1") as df_input:
        df = pd.read_csv(df_input)

    with open(working_dir + "uperl_und_liu_" + file, 'r', encoding="ISO-8859-1") as file_input, \
            open(result_dir + "null_field_file.csv", 'a+', encoding="utf-8", newline='') as file_output:

        reader = csv.reader(file_input)
        writer = csv.writer(file_output)
        fieldnames = next(reader)  # 获取数据的第一行，作为后续要转为字典的键名生成器，next方法获取
        print("the field names of input file are ", fieldnames)
        print("the type of field names is ", type(fieldnames))

        # sort out _liu field names and null columns
        null_field = []
        null_field.append(file)
        for field_name in fieldnames:

            # remove the null columns
            if df[field_name].count() == 0:
                null_field.append(field_name)
                df.drop(columns=[field_name], inplace=True)

            # remove the column _liu
            if field_name[-4:] == "_liu":
                df.drop(columns=[field_name], inplace=True)

        df.to_csv(result_dir + file, encoding="ISO-8859-1", index=False, mode='a')
        writer.writerow(null_field)



