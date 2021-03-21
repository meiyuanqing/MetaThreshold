#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/3/21
Time: 9:57
File: remove_undefined.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

the script removes the undefined data of metrics.
input： the metrics data includes the undefined metrics values.---------trainings.
output: removes the undefined metrics values.----------trainingsData.
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

working_dir = "F:\\NJU\\MTmeta\\experiments\\testings\\"
# working_dir = "F:\\NJU\\MTmeta\\experiments\\trainings\\"
result_dir = "F:\\NJU\\MTmeta\\experiments\\testingData\\"
# result_dir = "F:\\NJU\\MTmeta\\experiments\\trainingData\\"

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

    # removes the undefined metric values to the undefined_file.csv.
    with open(working_dir +  file, 'r', encoding="ISO-8859-1") as file_input, \
            open(result_dir + "undefined_file.csv", 'a+', encoding="utf-8", newline='') as file_output:

        reader = csv.reader(file_input)
        writer = csv.writer(file_output)
        fieldnames = next(reader)  # 获取数据的第一行，作为后续要转为字典的键名生成器，next方法获取
        print("the field names of input file are ", fieldnames)
        print("the type of field names is ", type(fieldnames))

        # sort out field names including the undefined metric values.
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
                print("the field name is ", field_name, " the undef_sum is ", undef_sum, "the undefined_sum is ",
                      undefined_sum)

        df.to_csv(result_dir + file, encoding="ISO-8859-1", index=False, mode='a')
        writer.writerow(undefined_field)
        # break

