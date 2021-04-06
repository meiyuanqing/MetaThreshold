#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/3/31
Time: 10:12
File: TDtool_processed_Ferreira
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

working_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\FerreiraThreshold\\"
result_dir = "F:\\NJU\\MTmeta\\experiments\\unsupervised\\FerreiraThresholdMinified\\"


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

    df = pd.read_csv(working_dir + file[:-4] + "\\Ferreira's Method Output\\Final-Result.csv", sep=';',
                     keep_default_na=False)

    df['metric_label'] = df.apply(lambda x: x['Metric'] + "_" + x['Lable'], axis=1)
    Ferreira_Threshold_minified_fieldname = ["fileName"]
    for i in range(len(df['metric_label'])):
        Ferreira_Threshold_minified_fieldname.append(df.loc[i, 'metric_label'])

    # reads a csv file and stores as dataframe
    with open(working_dir + file[:-4] + "\\Ferreira's Method Output\\Final-Result.csv",
              'r', encoding="ISO-8859-1") as df_input, \
            open(result_dir + "Ferreira_Threshold_minified.csv", 'a+', encoding="utf-8", newline='') as file_output:

        writer = csv.writer(file_output)
        print("the size of file_output is ", os.path.getsize(result_dir + "Ferreira_Threshold_minified.csv"))
        if os.path.getsize(result_dir + "Ferreira_Threshold_minified.csv") == 0:
            writer.writerow(Ferreira_Threshold_minified_fieldname)

        df = pd.read_csv(df_input, sep=';', keep_default_na=False)
        df['metric_label'] = df.apply(lambda x: x['Metric'] + "_" + x['Lable'], axis=1)
        # print("the df is ", df['metric_label'])
        # df_metric_value = pd.merge(df['metric_label'] + df['Value'])
        # print("the transform of df is ", df.transpose())
        Ferreira_Threshold_minified_row = [file[:-4]]
        for i in range(len(df['metric_label'])):

            value = df.loc[i, 'Value']
            if value[0] == '>':
                value = value.replace(value[0], '')
                value = int(value) + 1
            elif value[0] == '<':
                value = value.replace(value[0], '')
                if int(value) > 0:
                    value = int(value) - 1
                else:
                    value = int(value)
            else:
                value = int(value.split('-')[0])

            print("the value of metric is ", value)
            if df.loc[i, 'Metric'] in decimal_metric:
                print("the number of i is ", i, " and the metric is ", df.loc[i, 'Metric'])
                print("the ", df.loc[i, 'metric_label'], " value is ", df.loc[i, 'Value'],
                      " and the minified 100 times value is ", value * 0.01)
                Ferreira_Threshold_minified_row.append(value * 0.01)
                continue

            Ferreira_Threshold_minified_row.append(value)

        writer.writerow(Ferreira_Threshold_minified_row)

        # break
