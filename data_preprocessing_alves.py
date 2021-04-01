#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/3/29
Time: 10:06
File: data_preprocessing_alves.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

数据预处理：
（1）文档第一字母是乱码，记得好像是存储格式问题；（先不管，要出现了，再解决）
（2）度量只能是大于零的数值，不能为字符型
（3）csv中只能用分号分隔
（4）Alves中必须有loc度量
At the same time, removes the metrics that all equal to zero.
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

# working_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\testings\\"
working_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\trainings\\"
# result_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\testingData\\"
result_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\trainingData\\"

os.chdir(working_dir)
print(os.getcwd())

with open(working_dir + "List.txt") as l:
    lines = l.readlines()

for line in lines:

    file = line.replace("\n", "")
    print(repr(line))
    print(file)

    # reads a csv file and stores as dataframe
    with open(working_dir + file, 'r', encoding="ISO-8859-1") as df_input:
        df = pd.read_csv(df_input)

    # 去掉含有缺失值的样本（行）
    # df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    # 去掉含有缺失值的样本（行）
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

    # removes the undefined metric values to the undefined_file.csv.
    with open(working_dir + file, 'r', encoding="ISO-8859-1") as file_input, \
            open(result_dir + "undefined_file.csv", 'a+', encoding="utf-8", newline='') as file_output:

        reader = csv.reader(file_input)
        writer = csv.writer(file_output)
        fieldnames = next(reader)  # 获取数据的第一行，作为后续要转为字典的键名生成器，next方法获取
        print("the field names of input file are ", fieldnames)
        print("the type of field names is ", type(fieldnames))
        # "relName","className","currsloc",“bug"
        non_metric = ["prevsloc", "addedsloc", "deletedsloc", "changedsloc", "totalChangedsloc", "Kind",
                      "Name", "File", "CAMC", "Co", "DCd", "DCi", "LCC", "LCOM1", "LCOM2", "LCOM3", "LCOM4", "LCOM5",
                      "NHD", "NewCo", "NewLCOM5", "OCC", "PCC", "SNHD", "TCC"]

        # sort out field names including the undefined metric values and all-zero metrics.
        undefined_field = []
        undefined_field.append(file)
        for field_name in fieldnames:

            # counts the undef or undefined metrics number of each field
            undef_sum = (df[field_name] == "undef").sum()
            undefined_sum = (df[field_name] == "undefined").sum()
            # undef = df[field_name].str.contains('undef').count()
            # undefined = df[field_name].str.contains('undefined').count()

            # remove the null columns if the field contains undef or undefined
            if undef_sum > 0 or undefined_sum > 0:
                undefined_field.append(field_name)
                df.drop(columns=[field_name], inplace=True)
                # print("the field name is ", field_name, " the undef_sum is ", undef_sum, "the undefined_sum is ",
                #       undefined_sum)

            # removes the non_metrics
            if (field_name in non_metric) and (field_name in df.columns.values.tolist()):
                undefined_field.append(field_name)
                df.drop(columns=[field_name], inplace=True)
                print("the non metric of ", field_name, " is deleted!")

            # removes the metrics with all zero values
            if (field_name in df.columns.values.tolist()) and (df[field_name].value_counts().count() <= 1):
                print("The metric ", field_name, " and its sum value is ", df[field_name].value_counts().count())
                undefined_field.append(field_name)
                df.drop(columns=[field_name], inplace=True)
                print("the all-zero metric of ", field_name, " is deleted!")


        df.to_csv(result_dir + file, encoding="ISO-8859-1", index=False, mode='a')
        writer.writerow(undefined_field)
        # break
