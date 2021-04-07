#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/3/31
Time: 10:12
File: TDtool_processed_Oliveira
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


working_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\OliveiraThreshold\\"
result_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\OliveiraThresholdMinified\\"


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

    df_decimal = pd.read_csv(working_dir + "decimal_metric_file.csv", encoding='utf-8', header=None)
    # print("the df_decimal is ", df_decimal[0])
    decimal_metric = []
    for i_decimal in range(len(df_decimal[0])):
        if df_decimal.loc[i_decimal, 0] == file:
            print("the df_decimal is ", df_decimal.loc[i_decimal, 0])
            decimal_metric = df_decimal.loc[i_decimal, :].tolist()
            print("the decimal metircs are ", decimal_metric)
    # 当sep = ','的默认值时，index_col=0的默认值为0，需要改为False这样不会把第一列作为行号。
    df = pd.read_csv(working_dir + file[:-4] + "\\FinalResult.csv", keep_default_na=False, index_col=False)
    # print("the columns of df is ", df.columns.tolist())
    # print(df.loc[61, 'percentile'] == '100.0%')
    df['metric_percentile'] = df.apply(lambda x: x['metric'] + "_" + x['percentile'], axis=1)
    Oliveira_Threshold_minified_fieldname = ["fileName"]
    for i in range(len(df['metric_percentile'])):
        Oliveira_Threshold_minified_fieldname.append(df.loc[i, 'metric_percentile'])

    # reads a csv file and stores as dataframe
    with open(working_dir + file[:-4] + "\\FinalResult.csv", 'r', encoding="ISO-8859-1") as df_input, \
            open(result_dir + "Oliveira_Threshold_minified.csv", 'a+', encoding="utf-8", newline='') as file_output:

        writer = csv.writer(file_output)
        print("the size of file_output is ", os.path.getsize(result_dir + "Oliveira_Threshold_minified.csv"))
        if os.path.getsize(result_dir + "Oliveira_Threshold_minified.csv") == 0:
            writer.writerow(Oliveira_Threshold_minified_fieldname)

        df = pd.read_csv(df_input, sep=',', keep_default_na=False, index_col=False)
        df['metric_percentile'] = df.apply(lambda x: x['metric'] + "_" + x['percentile'], axis=1)

        Oliveira_Threshold_minified_row = [file[:-4]]
        for i in range(len(df['metric_percentile'])):

            if df.loc[i, 'percentile'] == '100.0%':
                k_value = df.loc[i, 'k']
            else:
                k_value = df.loc[i, 'k'] + 1

            if df.loc[i, 'metric'] in decimal_metric:
                print("the number of i is ", i, " and the metric is ", df.loc[i, 'metric'])
                print("the ", df.loc[i, 'metric_percentile'], " value is ", k_value,
                      " and the minified 100 times value is ", k_value * 0.01)
                Oliveira_Threshold_minified_row.append(k_value * 0.01)
                continue
            Oliveira_Threshold_minified_row.append(k_value)

        writer.writerow(Oliveira_Threshold_minified_row)

        # break
