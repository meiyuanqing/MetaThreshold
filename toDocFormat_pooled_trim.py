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
    31个规模度量(表A1)、18个复杂性度量(表A2)、20个耦合性度量(表A3)、19个继承性度量(表A4)和两个内聚性度量(表A5)，一共90个度量。

"""
import time


def doc_format(dir_file):
    import os
    import xlwt
    import pandas as pd

    # read_csv(path, keep_default_na=False, na_values=[""])  只有一个空字段将被识别为NaN
    df = pd.read_csv(dir_file, keep_default_na=False, na_values=[""])
    # 把metric列中有下划线的后部分去掉
    for k in range(len(df)):
        metric_name = df.loc[k, "metric"]
        metric_name = metric_name.split("_")[0]
        df.loc[k, "metric"] = metric_name

    file = dir_file.split("\\")[-1]
    directory = dir_file.replace(file, "")
    os.chdir(directory)

    workbook = xlwt.Workbook(encoding='utf-8')
    # 阈值	方差	P值	[LL, 	UL]	τ	Q	p-value	I2	[LL_Pred,	UL_Pred]
    columns = ["metric", "k_0", "direction", file[:-5] + "_adjusted", file[:-5] + "_stdError_adjusted",
               "pValue_Z_adjusted", "LL_CI_adjusted", "UL_CI_adjusted", "tau_adjusted", "Q_adjusted",
               "pValue_Q_adjusted", "I2_adjusted", "LL_ndPred_adjusted", "UL_ndPred_adjusted"]
    # columns = ["metric", file[:-5], file[:-5] + "_stdError", "pValue_Z", "LL_CI", "UL_CI", "tau", "Q",  "pValue_Q",
    #            "I2", "LL_ndPred", "UL_tdPred"]
    print(columns)
    # size metric
    booksheet_size = workbook.add_sheet("size", cell_overwrite_ok=True)
    size = ["AvgLine", "AvgLineBlank", "AvgLineComment", "AvgSLOC", "CountClassBase", "CountDeclClassVariable",
            "CountDeclInstanceVariable", "CountDeclMethodDefault", "CountDeclMethodPrivate", "CountDeclMethodProtected",
            "CountLine", "CountLineBlank", "CountLineCodeDecl", "CountLineCodeExe", "CountLineComment",
            "CountSemicolon", "CountStmtDecl", "CountStmtExe", "NA", "NAIMP", "NCM", "NIM", "NLM", "NM", "NMIMP",
            "NMNpub", "Nmpub", "NTM", "NumPara", "SLOC", "stms"]
    for j in range(len(columns)):
        booksheet_size.write(0, j, columns[j])

    for i in range(len(size) + 1):
        if i == 0:
            continue
        print(i - 1, size[i - 1])
        if df[df["metric"] == size[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0] > \
                df[df["metric"] == size[i - 1]].loc[:, file[:-5]].values[0]:
            direction = "Right"
        else:
            direction = "Left"
        booksheet_size.write(i, 0, size[i - 1])
        # print("the type is ",  df[df["metric"] == size[i-1]].loc[:, file[:-5]])
        # print("the type is ",  df[df["metric"] == size[i-1]].loc[:, file[:-5]].values)
        # print("the type is ",  df[df["metric"] == size[i-1]].loc[:, file[:-5]].values[0])
        # print("the type is ",  round(df[df["metric"] == size[i-1]].loc[:, file[:-5]].values[0], 3))
        # print("the type is ",  repr(round(df[df["metric"] == size[i-1]].loc[:, file[:-5]].values[0], 3)))
        # print("the type is ",  type(df[df["metric"] == size[i-1]].loc[:, file[:-5]].values[0]))
        # meta is -5, pearson is -4
        booksheet_size.write(i, 1, str(df[df["metric"] == size[i - 1]].loc[:, "k_0"].values[0]))
        booksheet_size.write(i, 2, direction)
        booksheet_size.write(i, 3, round(df[df["metric"] == size[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0], 3))
        booksheet_size.write(i, 4, round(df[df["metric"] == size[i - 1]].loc[:, file[:-5] + "_stdError_adjusted"].astype(float).values[0], 3))
        booksheet_size.write(i, 5, round(df[df["metric"] == size[i - 1]].loc[:, "pValue_Z_adjusted"].astype(float).values[0], 3))
        booksheet_size.write(i, 6, "[" + str(round(df[df["metric"] == size[i - 1]].loc[:, "LL_CI_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_size.write(i, 7, str(round(df[df["metric"] == size[i - 1]].loc[:, "UL_CI_adjusted"].astype(float).values[0], 3)) + "]")
        booksheet_size.write(i, 8, round(df[df["metric"] == size[i - 1]].loc[:, "tau_adjusted"].astype(float).values[0], 3))
        booksheet_size.write(i, 9, round(df[df["metric"] == size[i - 1]].loc[:, "Q_adjusted"].astype(float).values[0], 3))
        booksheet_size.write(i, 10, round(df[df["metric"] == size[i - 1]].loc[:, "pValue_Q_adjusted"].astype(float).values[0], 3))
        booksheet_size.write(i, 11, round(df[df["metric"] == size[i - 1]].loc[:, "I2_adjusted"].astype(float).values[0], 3))
        booksheet_size.write(i, 12, "[" + str(round(df[df["metric"] == size[i - 1]].loc[:, "LL_ndPred_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_size.write(i, 13, str(round(df[df["metric"] == size[i - 1]].loc[:, "UL_ndPred_adjusted"].astype(float).values[0], 3)) + "]")

    # complexity metric
    booksheet_complexity = workbook.add_sheet("complexity", cell_overwrite_ok=True)
    complexity = ["AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "AvgEssential", "CCMax", "CDE",
                  "CIE", "MaxCyclomaticModified", "MaxCyclomaticStrict", "MaxEssential", "MaxNesting",
                  "RatioCommentToCode", "SDMC", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential", "WMC",
                  "AvgWMC"]
    for j in range(len(columns)):
        booksheet_complexity.write(0, j, columns[j])
    for i in range(len(complexity) + 1):
        if i == 0:
            continue
        if df[df["metric"] == complexity[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0] > \
                df[df["metric"] == complexity[i - 1]].loc[:, file[:-5]].values[0]:
            direction = "Right"
        else:
            direction = "Left"
        print(i - 1, complexity[i - 1])
        booksheet_complexity.write(i, 0, complexity[i - 1])
        booksheet_complexity.write(i, 1, str(df[df["metric"] == complexity[i - 1]].loc[:, "k_0"].values[0]))
        booksheet_complexity.write(i, 2, direction)
        booksheet_complexity.write(i, 3, round(df[df["metric"] == complexity[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0], 3))
        booksheet_complexity.write(i, 4, round(df[df["metric"] == complexity[i - 1]].loc[:, file[:-5] + "_stdError_adjusted"].astype(float).values[0], 3))
        booksheet_complexity.write(i, 5, round(df[df["metric"] == complexity[i - 1]].loc[:, "pValue_Z_adjusted"].astype(float).values[0], 3))
        booksheet_complexity.write(i, 6, "[" + str(round(df[df["metric"] == complexity[i - 1]].loc[:, "LL_CI_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_complexity.write(i, 7, str(round(df[df["metric"] == complexity[i - 1]].loc[:, "UL_CI_adjusted"].astype(float).values[0], 3)) + "]")
        booksheet_complexity.write(i, 8, round(df[df["metric"] == complexity[i - 1]].loc[:, "tau_adjusted"].astype(float).values[0], 3))
        booksheet_complexity.write(i, 9, round(df[df["metric"] == complexity[i - 1]].loc[:, "Q_adjusted"].astype(float).values[0], 3))
        booksheet_complexity.write(i, 10, round(df[df["metric"] == complexity[i - 1]].loc[:, "pValue_Q_adjusted"].astype(float).values[0], 3))
        booksheet_complexity.write(i, 11, round(df[df["metric"] == complexity[i - 1]].loc[:, "I2_adjusted"].astype(float).values[0], 3))
        booksheet_complexity.write(i, 12, "[" + str(round(df[df["metric"] == complexity[i - 1]].loc[:, "LL_ndPred_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_complexity.write(i, 13, str(round(df[df["metric"] == complexity[i - 1]].loc[:, "UL_ndPred_adjusted"].astype(float).values[0], 3)) + "]")

    # Coupling metric
    booksheet_coupling = workbook.add_sheet("coupling", cell_overwrite_ok=True)
    coupling = ["ACAIC", "ACMIC", "AMMIC", "CBI", "CBO", "CountDeclMethodAll", "DAC", "DACquote", "DMMEC", "ICP",
                "IHICP", "MPC", "NIHICP", "OCAEC", "OCAIC", "OCMEC", "OCMIC", "OMMEC", "OMMIC", "RFC"]
    for j in range(len(columns)):
        booksheet_coupling.write(0, j, columns[j])
    for i in range(len(coupling) + 1):
        if i == 0:
            continue
        print(i - 1, coupling[i - 1])
        if df[df["metric"] == coupling[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0] > \
                df[df["metric"] == coupling[i - 1]].loc[:, file[:-5]].values[0]:
            direction = "Right"
        else:
            direction = "Left"
        booksheet_coupling.write(i, 0, coupling[i - 1])
        booksheet_coupling.write(i, 1, str(df[df["metric"] == coupling[i - 1]].loc[:, "k_0"].values[0]))
        booksheet_coupling.write(i, 2, direction)
        booksheet_coupling.write(i, 3, round(df[df["metric"] == coupling[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0], 3))
        booksheet_coupling.write(i, 4, round(df[df["metric"] == coupling[i - 1]].loc[:, file[:-5] + "_stdError_adjusted"].astype(float).values[0], 3))
        booksheet_coupling.write(i, 5, round(df[df["metric"] == coupling[i - 1]].loc[:, "pValue_Z_adjusted"].astype(float).values[0], 3))
        booksheet_coupling.write(i, 6, "[" + str(round(df[df["metric"] == coupling[i - 1]].loc[:, "LL_CI_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_coupling.write(i, 7, str(round(df[df["metric"] == coupling[i - 1]].loc[:, "UL_CI_adjusted"].astype(float).values[0], 3)) + "]")
        booksheet_coupling.write(i, 8, round(df[df["metric"] == coupling[i - 1]].loc[:, "tau_adjusted"].astype(float).values[0], 3))
        booksheet_coupling.write(i, 9, round(df[df["metric"] == coupling[i - 1]].loc[:, "Q_adjusted"].astype(float).values[0], 3))
        booksheet_coupling.write(i, 10, round(df[df["metric"] == coupling[i - 1]].loc[:, "pValue_Q_adjusted"].astype(float).values[0], 3))
        booksheet_coupling.write(i, 11, round(df[df["metric"] == coupling[i - 1]].loc[:, "I2_adjusted"].astype(float).values[0], 3))
        booksheet_coupling.write(i, 12, "[" + str(round(df[df["metric"] == coupling[i - 1]].loc[:, "LL_ndPred_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_coupling.write(i, 13, str(round(df[df["metric"] == coupling[i - 1]].loc[:, "UL_ndPred_adjusted"].astype(float).values[0], 3)) + "]")

    # Inheritance metric "PII",  "SIX"(在"PII"),"SPD"(最后一个)
    booksheet_inheritance = workbook.add_sheet("inheritance", cell_overwrite_ok=True)
    inheritance = ["AID", "CLD", "DIT", "DP", "DPA", "DPD", "MaxInheritanceTree", "NMA", "NMI", "NMO", "NOA", "NOC",
                   "NOD", "NOP", "PII", "SIX", "SP", "SPA"]
    for j in range(len(columns)):
        booksheet_inheritance.write(0, j, columns[j])
    for i in range(len(inheritance) + 1):
        if i == 0:
            continue
        print(i - 1, inheritance[i - 1])
        if df[df["metric"] == inheritance[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0] > \
                df[df["metric"] == inheritance[i - 1]].loc[:, file[:-5]].values[0]:
            direction = "Right"
        else:
            direction = "Left"
        booksheet_inheritance.write(i, 0, inheritance[i - 1])
        booksheet_inheritance.write(i, 1, str(df[df["metric"] == inheritance[i - 1]].loc[:, "k_0"].values[0]))
        booksheet_inheritance.write(i, 2, direction)
        booksheet_inheritance.write(i, 3, round(df[df["metric"] == inheritance[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0], 3))
        booksheet_inheritance.write(i, 4, round(df[df["metric"] == inheritance[i - 1]].loc[:, file[:-5] + "_stdError_adjusted"].astype(float).values[0], 3))
        booksheet_inheritance.write(i, 5, round(df[df["metric"] == inheritance[i - 1]].loc[:, "pValue_Z_adjusted"].astype(float).values[0], 3))
        booksheet_inheritance.write(i, 6, "[" + str(round(df[df["metric"] == inheritance[i - 1]].loc[:, "LL_CI_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_inheritance.write(i, 7, str(round(df[df["metric"] == inheritance[i - 1]].loc[:, "UL_CI_adjusted"].astype(float).values[0], 3)) + "]")
        booksheet_inheritance.write(i, 8, round(df[df["metric"] == inheritance[i - 1]].loc[:, "tau_adjusted"].astype(float).values[0], 3))
        booksheet_inheritance.write(i, 9, round(df[df["metric"] == inheritance[i - 1]].loc[:, "Q_adjusted"].astype(float).values[0], 3))
        booksheet_inheritance.write(i, 10, round(df[df["metric"] == inheritance[i - 1]].loc[:, "pValue_Q_adjusted"].astype(float).values[0], 3))
        booksheet_inheritance.write(i, 11, round(df[df["metric"] == inheritance[i - 1]].loc[:, "I2_adjusted"].astype(float).values[0], 3))
        booksheet_inheritance.write(i, 12, "[" + str(round(df[df["metric"] == inheritance[i - 1]].loc[:, "LL_ndPred_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_inheritance.write(i, 13, str(round(df[df["metric"] == inheritance[i - 1]].loc[:, "UL_ndPred_adjusted"].astype(float).values[0], 3)) + "]")

    # Cohesion metric
    booksheet_cohesion = workbook.add_sheet("cohesion", cell_overwrite_ok=True)
    cohesion = ["ICH", "PercentLackOfCohesion"]
    for j in range(len(columns)):
        booksheet_cohesion.write(0, j, columns[j])
    for i in range(len(cohesion) + 1):
        if i == 0:
            continue
        print(i - 1, cohesion[i - 1])
        if df[df["metric"] == cohesion[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0] > \
                df[df["metric"] == cohesion[i - 1]].loc[:, file[:-5]].values[0]:
            direction = "Right"
        else:
            direction = "Left"
        booksheet_cohesion.write(i, 0, cohesion[i - 1])
        booksheet_cohesion.write(i, 1, str(df[df["metric"] == cohesion[i - 1]].loc[:, "k_0"].values[0]))
        booksheet_cohesion.write(i, 2, direction)
        booksheet_cohesion.write(i, 3, round(df[df["metric"] == cohesion[i - 1]].loc[:, file[:-5] + "_adjusted"].astype(float).values[0], 3))
        booksheet_cohesion.write(i, 4, round(df[df["metric"] == cohesion[i - 1]].loc[:, file[:-5] + "_stdError_adjusted"].astype(float).values[0], 3))
        booksheet_cohesion.write(i, 5, round(df[df["metric"] == cohesion[i - 1]].loc[:, "pValue_Z_adjusted"].astype(float).values[0], 3))
        booksheet_cohesion.write(i, 6, "[" + str(round(df[df["metric"] == cohesion[i - 1]].loc[:, "LL_CI_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_cohesion.write(i, 7, str(round(df[df["metric"] == cohesion[i - 1]].loc[:, "UL_CI_adjusted"].astype(float).values[0], 3)) + "]")
        booksheet_cohesion.write(i, 8, round(df[df["metric"] == cohesion[i - 1]].loc[:, "tau_adjusted"].astype(float).values[0], 3))
        booksheet_cohesion.write(i, 9, round(df[df["metric"] == cohesion[i - 1]].loc[:, "Q_adjusted"].astype(float).values[0], 3))
        booksheet_cohesion.write(i, 10, round(df[df["metric"] == cohesion[i - 1]].loc[:, "pValue_Q_adjusted"].astype(float).values[0], 3))
        booksheet_cohesion.write(i, 11, round(df[df["metric"] == cohesion[i - 1]].loc[:, "I2_adjusted"].astype(float).values[0], 3))
        booksheet_cohesion.write(i, 12, "[" + str(round(df[df["metric"] == cohesion[i - 1]].loc[:, "LL_ndPred_adjusted"].astype(float).values[0], 3)) + ",")
        booksheet_cohesion.write(i, 13, str(round(df[df["metric"] == cohesion[i - 1]].loc[:, "UL_ndPred_adjusted"].astype(float).values[0], 3)) + "]")

    workbook.save(directory + "doc_trim_" + file.replace("csv", "xls"))


if __name__ == '__main__':

    s_time = time.time()
    # m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled\\Pooled_meta_thresholds.csv"
    # m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled_all\\Pooled_meta_thresholds.csv"
    # m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled\\AUC_Pooled_metas.csv"
    # m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled\\AUC_Alves_metas.csv"
    m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled_all\\single_method\\supervised_meta_thresholds.csv"

    doc_format(m_dir_file)
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of toDocFormat.py!\n",
          "The execution time of toDocFormat.py script is ", execution_time)
