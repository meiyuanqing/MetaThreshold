#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/3/29
Time: 19:47
File: magnify100.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

For the TDtool does not derive the threshold for a decimal value, we magnify a decimal metric by 100 times.

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
working_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\trainingData\\"
# result_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\testingData\\"
result_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\trainingDataMagnified\\"

os.chdir(working_dir)
print(os.getcwd())

with open(working_dir + "List.txt") as l:
    lines = l.readlines()

for line in lines:

    file = line.replace("\n", "")
    print(repr(line))
    print(file)

    # creates the folder according to file name
    if not os.path.exists(result_dir + file[:-4]):
        os.makedirs(result_dir + file[:-4])

    # reads a csv file and stores as dataframe
    with open(working_dir + file, 'r', encoding="ISO-8859-1") as df_input:
        df = pd.read_csv(df_input)

    # 去掉含有缺失值的样本（行）
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    print("the types of df is ", df.dtypes)

    # removes the undefined metric values to the undefined_file.csv.
    with open(working_dir + file, 'r', encoding="ISO-8859-1") as file_input, \
            open(result_dir + "decimal_metric_file.csv", 'a+', encoding="utf-8", newline='') as file_output:

        reader = csv.reader(file_input)
        writer = csv.writer(file_output)
        fieldnames = next(reader)  # 获取数据的第一行，作为后续要转为字典的键名生成器，next方法获取
        print("the field names of input file are ", fieldnames)
        print("the type of field names is ", type(fieldnames))
        # "relName","className","currsloc",“bug"
        non_metric = ["prevsloc", "addedsloc", "deletedsloc", "changedsloc", "totalChangedsloc", "Kind",
                      "Name", "File", "CAMC", "Co", "DCd", "DCi", "LCC", "LCOM1", "LCOM2", "LCOM3", "LCOM4", "LCOM5",
                      "NHD", "NewCo", "NewLCOM5", "OCC", "PCC", "SNHD", "TCC"]

        # sort out field names including the undefined metric values.
        decimal_metric = []
        decimal_metric.append(file)
        for field_name in fieldnames:

            # if the value of a metric is a decimal, magnify it by 1000 times.
            # counts the . number of each field
            print("the field name is ", field_name,
                  "the type of field is ", df[field_name].dtype,
                  "the type of field is ", repr(df[field_name].dtype),
                  "the boolean equation is ", df[field_name].dtype == "float64")
            if df[field_name].dtype == "float64":

                decimal_point_sum = df[field_name].apply(
                    lambda x: 1 if float(str(x).split(".")[1]) > 0 else 0).sum()
                # decimal_point_sum_value = df[field_name].sum()
                print("the decimal_point_sum is ", decimal_point_sum)
                if decimal_point_sum > 0:
                    df[field_name] = df[field_name].multiply(100).round()
                    print("the magnified column type is ", df[field_name].dtype)
                    decimal_metric.append(field_name)

                # transform all float type to int
                df[field_name] = df[field_name].apply(lambda x: int(x))

            # break

        df.to_csv(result_dir + file, encoding="ISO-8859-1", index=False, mode='a', sep=';')
        writer.writerow(decimal_metric)
        # break
