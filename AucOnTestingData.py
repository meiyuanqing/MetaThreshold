#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/4/9
Time: 15:46
File: AucOnTestingData.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Performance metric: AUC
(1)tests the unsupervised and supervised deriving method's threshold respectively on testing data;
(2)tests the 9 methods' meta-analysis value of threshold on testing data.

"""
import time


def auc_testing(threshold_dir="F:\\NJU\\MTmeta\\experiments\\pooled\\PoolingThresholds\\",
                testing_dir="F:\\NJU\\MTmeta\\experiments\\supervised\\testingData\\",
                result_dir="F:\\NJU\\MTmeta\\experiments\\pooled\\",
                threshold_list="List_t.txt",
                testing_list="List.txt"):
    import os
    import csv
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    # from sklearn import metrics
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    result_directory = result_dir
    os.chdir(threshold_dir)

    # read one of file to get the metric names for meta-analysis
    # read_csv(path, keep_default_na=False, na_values=[""])  只有一个空字段将被识别为NaN
    df_metric_names = pd.read_csv(threshold_dir + "Pooled_meta_thresholds.csv", keep_default_na=False,
                                  na_values=[""])

    metric_names = sorted(set(df_metric_names.metric.values.tolist()))
    print("the metric_names are ", df_metric_names.columns.values.tolist())
    print("the metric_names are ", metric_names)
    print("the len metric_names are ", len(metric_names))

    with open(testing_dir + testing_list) as l:
        lines = l.readlines()

    with open(threshold_dir + threshold_list) as l_t:
        lines_t = l_t.readlines()

    print("the files are ", lines)
    print("the number of list files is ", len(lines))
    print("the files_t are ", lines_t)
    print("the number_t of list files is ", len(lines_t))

    for line in lines:
        file = line.replace("\n", "")
        print('the file is ', file)

        # 分别处理每一个项目: f1取出要被处理的项目;
        #                  f2:用于存储每一个项目的Spearman,pearson,FisherZ和variance
        #                  deletedList: 用于存储项目中某个度量样本数小于3，和pearson等于1的度量。
        with open(testing_dir + file, 'r', encoding="ISO-8859-1") as f1, \
                open(result_directory + "AUCs.csv", 'a+', encoding="utf-8", newline='') as f2, \
                open(result_directory + "deleted.csv", 'a+', encoding="utf-8", newline='') as deleted_file:

            reader = csv.reader(f1)
            writer = csv.writer(f2)
            writer_deleted = csv.writer(deleted_file)
            # receives the first line of a file and convert to dict generator
            fieldnames = next(reader)
            # exclude the non metric fields
            non_metric = ["relName", "className", "bug"]

            # metric_data stores the metric fields (102 items)
            def fun_1(m):
                return m if m not in non_metric else None

            metric_data = filter(fun_1, fieldnames)

            df = pd.read_csv(testing_dir + file)
            # drop all rows that have any NaN values,删除表中含有任何NaN的行,并重新设置行号
            df = df.dropna(axis=0, how='any', inplace=False).reset_index(drop=True)

            if os.path.getsize(result_directory + "AUCs.csv") == 0:
                writer.writerow(["fileName", "metric", "Sample_size",
                                 "Threshold_Alves", "AUC_Alves", "AUC_Alves_variance",
                                 "Threshold_bpp", "AUC_bpp", "AUC_bpp_variance",
                                 "Threshold_Ferreira", "AUC_Ferreira", "AUC_Ferreira_variance",
                                 "Threshold_gm", "AUC_gm", "AUC_gm_variance",
                                 "Threshold_mfm", "AUC_mfm", "AUC_mfm_variance",
                                 "Threshold_Oliveira", "AUC_Oliveira", "AUC_Oliveira_variance",
                                 "Threshold_roc", "AUC_roc", "AUC_roc_variance",
                                 "Threshold_Vale", "AUC_Vale", "AUC_Vale_variance",
                                 "Threshold_varl", "AUC_varl", "AUC_varl_variance",
                                 "Threshold_Pooled", "AUC_Pooled", "AUC_Pooled_variance"])

            if os.path.getsize(result_directory + "deleted.csv") == 0:
                writer_deleted.writerow(["fileName"])

            if len(df) <= 6:
                writer_deleted.writerow([file])
                continue

            for metric in metric_names:
                print("the current file is ", file)
                print("the current metric is ", metric)

                metric_row = [file, metric, len(df)]
                # 由于bug中存储的是缺陷个数,转化为二进制存储,若x>2,则可预测bug为3个以上的阈值,其他类推
                df['bugBinary'] = df.bug.apply(lambda x: 1 if x > 0 else 0)
                # reads the thresholds from 10 threshold files
                for line_t in lines_t:
                    file_t = line_t.replace("\n", "")
                    print('the file is ', file_t)
                    print('the threshold column name is ', file_t[:-5])
                    method_name = file_t.split("_")[0]
                    print("the method is ", method_name)
                    df_t = pd.read_csv(file_t, keep_default_na=False, na_values=[""])
                    method_name_t = float(df_t[df_t["metric"] == metric].loc[:, file_t[:-5]].values[0])
                    df['predictBinary'] = df[metric].apply(lambda x: 1 if x >= method_name_t else 0)
                    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
                    c_matrix = confusion_matrix(df["bugBinary"], df['predictBinary'], labels=[0, 1])
                    auc_value = roc_auc_score(df['bugBinary'], df['predictBinary'])
                    valueOfbugBinary = df["predictBinary"].value_counts()  # 0 和 1 的各自的个数

                    print("the auc_value of df is ", auc_value)
                    print("the value of df is ", valueOfbugBinary)
                    print("the type value of df is ", type(valueOfbugBinary))
                    print("the repr value of df is ", repr(valueOfbugBinary))
                    print("the index value of df is ", valueOfbugBinary.keys())
                    # print("the index value of df is ", valueOfbugBinary.keys()[0])
                    # print("the index value of df is ", valueOfbugBinary.keys()[1])
                    # print("the value of valueOfbugBinary[0] is ", valueOfbugBinary[0])
                    # print("the value of valueOfbugBinary[1] is ", valueOfbugBinary[1])
                    if len(valueOfbugBinary) <= 1:
                        if valueOfbugBinary.keys()[0] == 0:
                            value_0 = valueOfbugBinary[0]
                            value_1 = 0
                        else:
                            value_0 = 0
                            value_1 = valueOfbugBinary[1]
                    else:
                        value_0 = valueOfbugBinary[0]
                        value_1 = valueOfbugBinary[1]
                    print("the value_0 is ", value_0, "the value_1 is ", value_1)
                    Q1 = auc_value / (2 - auc_value)
                    Q2 = 2 * auc_value * auc_value / (1 + auc_value)
                    auc_value_variance = auc_value * (1 - auc_value) \
                                         + (value_1 - 1) * (Q1 - auc_value * auc_value)\
                                         + (value_0 - 1) * (Q2 - auc_value * auc_value)
                    auc_value_variance = auc_value_variance / (value_0 * value_1)
                    metric_row.append(method_name_t)
                    metric_row.append(auc_value)
                    metric_row.append(auc_value_variance)
                    print("the method is ", method_name, " and the auc is ", auc_value,
                          " the variance is ", auc_value_variance)

                writer.writerow(metric_row)
                # break


if __name__ == '__main__':
    s_time = time.time()
    auc_testing()
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of AucOnTestingData.py!\n",
          "The execution time of AucOnTestingData.py script is ", execution_time)
