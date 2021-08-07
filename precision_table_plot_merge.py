#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/8/4
Time: 20:06
File: precision_table_plot.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

compute the avg std max min values and draw the box plot of precision and recall.
"""
import time


def precision_table_plot(working_dir="F:\\NJU\\MTmeta\\experiments\\pooled\\",
                         plot_dir="F:\\NJU\\MTmeta\\experiments\\pooled\\plots\\"):
    import os
    import csv
    import pandas as pd
    import matplotlib.pyplot as plt

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    plt.rcParams['savefig.dpi'] = 900  # 图片像素
    plt.rcParams['figure.figsize'] = (8.0, 4.0)

    os.chdir(working_dir)

    df = pd.read_csv(working_dir + "AUCs.csv", keep_default_na=False, na_values=[""])

    # df = df.drop(axis=0, how='any', inplace=False).reset_index(drop=True)

    metric_names = sorted(set(df.metric.values.tolist()))
    print("the metric_names are ", df.columns.values.tolist())
    print("the metric_names are ", metric_names)
    print("the len metric_names are ", len(metric_names))

    with open(working_dir + "precision_table.csv", 'a+', encoding="utf-8", newline='') as talbe:

        writer = csv.writer(talbe)
        if os.path.getsize(working_dir + "precision_table.csv") == 0:
            writer.writerow(["metric", "Sample_size", "recall_max", "recall_min", "recall_median", "recall_mean",
                             "recall_variance", "precision_max", "precision_min", "precision_median", "precision_mean",
                             "precision_variance", "f1_max", "f1_min", "f1_median", "f1_mean", "f1_variance"])

        # 需要把同类型所有的度量的性能指标画在一张图上，定义一个空数据框，用于存入行数相同的度量性能结果
        size = ["AvgLine", "AvgLineBlank", "AvgLineComment", "AvgSLOC", "CountClassBase", "CountDeclClassVariable",
                "CountDeclInstanceVariable", "CountDeclMethodDefault", "CountDeclMethodPrivate",
                "CountDeclMethodProtected",
                "CountLine", "CountLineBlank", "CountLineCodeDecl", "CountLineCodeExe", "CountLineComment",
                "CountSemicolon", "CountStmtDecl", "CountStmtExe", "NA", "NAIMP", "NCM", "NIM", "NLM", "NM", "NMIMP",
                "NMNpub", "Nmpub", "NTM", "NumPara", "SLOC", "stms"]

        complexity = ["AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "AvgEssential", "CCMax", "CDE",
                      "CIE", "MaxCyclomaticModified", "MaxCyclomaticStrict", "MaxEssential", "MaxNesting",
                      "RatioCommentToCode", "SDMC", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential",
                      "WMC", "AvgWMC"]

        coupling = ["ACAIC", "ACMIC", "AMMIC", "CBI", "CBO", "CountDeclMethodAll", "DAC", "DACquote", "DMMEC", "ICP",
                    "IHICP", "MPC", "NIHICP", "OCAEC", "OCAIC", "OCMEC", "OCMIC", "OMMEC", "OMMIC", "RFC"]

        inheritance = ["AID", "CLD", "DIT", "DP", "DPA", "DPD", "MaxInheritanceTree", "NMA", "NMI", "NMO", "NOA", "NOC",
                       "NOD", "NOP", "SIX", "SP", "SPA", "SPD", "ICH", "PercentLackOfCohesion"]

        # cohesion = ["ICH", "PercentLackOfCohesion"]

        size_recall_df = pd.DataFrame(columns=size)
        size_precision_df = pd.DataFrame(columns=size)
        complexity_recall_df = pd.DataFrame(columns=complexity)
        complexity_precision_df = pd.DataFrame(columns=complexity)
        coupling_recall_df = pd.DataFrame(columns=coupling)
        coupling_precision_df = pd.DataFrame(columns=coupling)
        inheritance_recall_df = pd.DataFrame(columns=inheritance)
        inheritance_precision_df = pd.DataFrame(columns=inheritance)

        for metric in metric_names:

            print("the current metric is ", metric)
            metric_row = [metric, len(df[df["metric"] == metric].loc[:, "recall_Pooled"])]
            # 由于bug中存储的是缺陷个数,转化为二进制存储,若x>2,则可预测bug为3个以上的阈值,其他类推

            metric_row.append(df[df["metric"] == metric].loc[:, "recall_Pooled"].max())
            metric_row.append(df[df["metric"] == metric].loc[:, "recall_Pooled"].min())
            metric_row.append(df[df["metric"] == metric].loc[:, "recall_Pooled"].median())
            metric_row.append(df[df["metric"] == metric].loc[:, "recall_Pooled"].mean())
            metric_row.append(df[df["metric"] == metric].loc[:, "recall_Pooled"].var())

            metric_row.append(df[df["metric"] == metric].loc[:, "precision_Pooled"].max())
            metric_row.append(df[df["metric"] == metric].loc[:, "precision_Pooled"].min())
            metric_row.append(df[df["metric"] == metric].loc[:, "precision_Pooled"].median())
            metric_row.append(df[df["metric"] == metric].loc[:, "precision_Pooled"].mean())
            metric_row.append(df[df["metric"] == metric].loc[:, "precision_Pooled"].var())

            metric_row.append(df[df["metric"] == metric].loc[:, "f1_Pooled"].max())
            metric_row.append(df[df["metric"] == metric].loc[:, "f1_Pooled"].min())
            metric_row.append(df[df["metric"] == metric].loc[:, "f1_Pooled"].median())
            metric_row.append(df[df["metric"] == metric].loc[:, "f1_Pooled"].mean())
            metric_row.append(df[df["metric"] == metric].loc[:, "f1_Pooled"].var())

            print("the mean value of recall is ", df[df["metric"] == metric].loc[:, "recall_Pooled"].mean())
            writer.writerow(metric_row)

            if metric in size:
                size_recall_df[metric] = df[df["metric"] == metric].loc[:, "recall_Pooled"].dropna(axis=0,
                                  how='any', inplace=False).reset_index(drop=True)
                size_precision_df[metric] = df[df["metric"] == metric].loc[:, "precision_Pooled"].dropna(axis=0,
                                  how='any', inplace=False).reset_index(drop=True)
            if metric in complexity:
                complexity_recall_df[metric] = df[df["metric"] == metric].loc[:, "recall_Pooled"].dropna(axis=0,
                                  how='any', inplace=False).reset_index(drop=True)
                complexity_precision_df[metric] = df[df["metric"] == metric].loc[:, "precision_Pooled"].dropna(axis=0,
                                  how='any', inplace=False).reset_index(drop=True)
            if metric in coupling:
                coupling_recall_df[metric] = df[df["metric"] == metric].loc[:, "recall_Pooled"].dropna(axis=0,
                                  how='any', inplace=False).reset_index(drop=True)
                coupling_precision_df[metric] = df[df["metric"] == metric].loc[:, "precision_Pooled"].dropna(axis=0,
                                  how='any', inplace=False).reset_index(drop=True)
            if metric in inheritance:
                inheritance_recall_df[metric] = df[df["metric"] == metric].loc[:, "recall_Pooled"].dropna(axis=0,
                                  how='any', inplace=False).reset_index(drop=True)
                inheritance_precision_df[metric] = df[df["metric"] == metric].loc[:, "precision_Pooled"].dropna(axis=0,
                                  how='any', inplace=False).reset_index(drop=True)

        print(size_recall_df)
        plt.rcParams['savefig.dpi'] = 900  # 图片像素
        plt.rcParams['figure.figsize'] = (20.0, 6.0)
        size_recall_df.plot.box(title="Recall values' Box Plot of Size Metrics")
        plt.grid(linestyle="--", alpha=0.3)
        # plt.xticks(rotation=0, fontsize=9.0)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.34) #这是增加下面标签的显示宽度，20210807找了下午5个小时
        plt.savefig(plot_dir + 'SizeRecallMetrics.png')
        plt.close()

        size_precision_df.plot.box(title="Precision values' Box Plot of Size Metrics")
        plt.grid(linestyle="--", alpha=0.3)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.34)
        plt.savefig(plot_dir + 'SizePrecisionMetrics.png')
        plt.close()

        complexity_recall_df.plot.box(title="Recall values' Box Plot of Complexity Metrics")
        plt.grid(linestyle="--", alpha=0.3)
        # plt.xticks(rotation=0, fontsize=9.0)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.34) #这是增加下面标签的显示宽度，20210807找了下午5个小时
        plt.savefig(plot_dir + 'complexityRecallMetrics.png')
        plt.close()

        complexity_precision_df.plot.box(title="Precision values' Box Plot of Complexity Metrics")
        plt.grid(linestyle="--", alpha=0.3)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.34)
        plt.savefig(plot_dir + 'complexityPrecisionMetrics.png')
        plt.close()

        coupling_recall_df.plot.box(title="Recall values' Box Plot of Coupling Metrics")
        plt.grid(linestyle="--", alpha=0.3)
        # plt.xticks(rotation=0, fontsize=9.0)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.34) #这是增加下面标签的显示宽度，20210807找了下午5个小时
        plt.savefig(plot_dir + 'couplingRecallMetrics.png')
        plt.close()

        coupling_precision_df.plot.box(title="Precision values' Box Plot of Coupling Metrics")
        plt.grid(linestyle="--", alpha=0.3)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.34)
        plt.savefig(plot_dir + 'couplingPrecisionMetrics.png')
        plt.close()

        inheritance_recall_df.plot.box(title="Recall values' Box Plot of Inheritance and Cohesion Metrics")
        plt.grid(linestyle="--", alpha=0.3)
        # plt.xticks(rotation=0, fontsize=9.0)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.34) #这是增加下面标签的显示宽度，20210807找了下午5个小时
        plt.savefig(plot_dir + 'inheritanceCohesionRecallMetrics.png')
        plt.close()

        inheritance_precision_df.plot.box(title="Precision values' Box Plot of Inheritance and Cohesion Metrics")
        plt.grid(linestyle="--", alpha=0.3)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.34)
        plt.savefig(plot_dir + 'inheritanceCohesionPrecisionMetrics.png')
        plt.close()

        # plt.boxplot([df[df["metric"] == metric].loc[:, "recall_Pooled"],
        #              df[df["metric"] == metric].loc[:, "precision_Pooled"]],
        #             showfliers=False, vert=False, labels=['recall', 'precision'],
        #             showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'indianred'})
        #
        # plt.title("recall and precision of " + metric)
        #
        # plt.savefig(plot_dir + "boxplot_" + metric + ".png")
        # plt.close()


if __name__ == '__main__':
    s_time = time.time()
    precision_table_plot()
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of precision_table_plot.py!\n",
          "The execution time of AucOnTestingData.py script is ", execution_time)
