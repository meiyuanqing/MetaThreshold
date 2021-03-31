#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/3/31
Time: 10:12
File: TDtool_processed
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

(1)Read the results from TDtool to csv files for meta-analysis;
(2)minify the magnified metrics by 100 times(9 items):SIX,CBI,RatioCommentToCode,AvgWMC,CDE,CIE,SDMC,AvgSLOC,PII.

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


working_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\ValeThreshold\\"

result_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\ValeThresholdMinified\\"

os.chdir(working_dir)
print(os.getcwd())

with open(working_dir + "List.txt") as l:
    lines = l.readlines()

print("the lines are ", lines)
print("the len of lines is ", len(lines))

for line in lines:

    file = line.replace("\n", "")
    print(repr(line))
    print(file)

    df_decimal = pd.read_csv(working_dir + "decimal_metric_file.csv", encoding='utf-8', header=-1)
    print("the types of df_decimal is ", df_decimal.dtypes)
    print("the df is ", df_decimal)

    # reads a csv file and stores as dataframe
    with open(working_dir + file[:-4] + "\\Vale's Method Output\\Final-Result.csv",
              'r', encoding="ISO-8859-1") as df_input, \
            open(working_dir + "decimal_metric_file.csv", 'r', encoding="utf-8", newline='') as decimal_input, \
            open(result_dir + "Vale_Threshold_minified.csv", 'a+', encoding="utf-8", newline='') as file_output:

        df = pd.read_csv(df_input, sep=';')

        print("the types of df is ", df.dtypes)
        print("the df is ", df)

        writer = csv.writer(file_output)

        # writer.writerow(decimal_metric)
        break