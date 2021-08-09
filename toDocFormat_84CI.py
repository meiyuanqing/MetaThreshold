#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/8/9
Time: 20:55
File: toDocFormat_84CI.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Read the original table to extract the 84CI and 84CI' values to doc.
When there is a overlap between 84CI and 84CI', it's robust to sensitivity analysis; otherwise, non-robust.

"""

import time


def doc_format(dir_file):
    import os
    import xlwt
    import pandas as pd

    # read_csv(path, keep_default_na=False, na_values=[""])  只有一个空字段将被识别为NaN
    df = pd.read_csv(dir_file, keep_default_na=False, na_values=[""])

    file = dir_file.split("\\")[-1]
    directory = dir_file.replace(file, "")
    os.chdir(directory)

    print("the file is ", file)

    workbook = xlwt.Workbook(encoding='utf-8')
    columns = ["metric", "LL_CI_84_trim", "UL_CI_84_trim", "LL_CI_84", "UL_CI_84"]
    print(columns)
    # size metric
    booksheet_size = workbook.add_sheet("size", cell_overwrite_ok=True)
    size = ["AvgLine", "AvgLineBlank", "AvgLineComment", "AvgSLOC", "CountClassBase", "CountDeclClassVariable",
            "CountDeclInstanceVariable", "CountDeclMethodDefault", "CountDeclMethodPrivate", "CountDeclMethodProtected",
            "CountLine", "CountLineBlank", "CountLineCodeDecl", "CountLineCodeExe", "CountLineComment", "CountSemicolon",
            "CountStmtDecl", "CountStmtExe", "NA", "NCM", "NIM", "NLM", "NM", "NMIMP", "NMNpub", "NumPara", "SLOC",
            "stms"]
    for j in range(len(columns)):
        booksheet_size.write(0, j, columns[j])

    for i in range(len(size) + 1):
        if i == 0:
            continue
        print(i - 1, size[i - 1])
        booksheet_size.write(i, 0, size[i - 1])
        booksheet_size.write(i, 1, "[" + str(round(df[df["metric"] == size[i - 1]].loc[:, "LL_CI_84_adjusted"].values[0], 3)) + ",")
        booksheet_size.write(i, 2, str(round(df[df["metric"] == size[i - 1]].loc[:, "UL_CI_84_adjusted"].values[0], 3)) + "]")
        booksheet_size.write(i, 3, "[" + str(round(df[df["metric"] == size[i - 1]].loc[:, "LL_CI_84"].values[0], 3)) + ",")
        booksheet_size.write(i, 4, str(round(df[df["metric"] == size[i - 1]].loc[:, "UL_CI_84"].values[0], 3)) + "]")

    # complexity metric
    booksheet_complexity = workbook.add_sheet("complexity", cell_overwrite_ok=True)
    complexity = ["AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict", "AvgEssential", "CDE",
                  "MaxCyclomaticStrict", "MaxEssential", "MaxNesting", "RatioCommentToCode", "SDMC",
                  "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential", "WMC", "AvgWMC"]

    for j in range(len(columns)):
        booksheet_complexity.write(0, j, columns[j])
    for i in range(len(complexity) + 1):
        if i == 0:
            continue
        print(i - 1, complexity[i - 1])
        booksheet_complexity.write(i, 0, complexity[i - 1])
        booksheet_complexity.write(i, 1, "[" + str(round(df[df["metric"] == complexity[i - 1]].loc[:, "LL_CI_84_adjusted"].values[0], 3)) + ",")
        booksheet_complexity.write(i, 2, str(round(df[df["metric"] == complexity[i - 1]].loc[:, "UL_CI_84_adjusted"].values[0], 3)) + "]")
        booksheet_complexity.write(i, 3, "[" + str(round(df[df["metric"] == complexity[i - 1]].loc[:, "LL_CI_84"].values[0], 3)) + ",")
        booksheet_complexity.write(i, 4, str(round(df[df["metric"] == complexity[i - 1]].loc[:, "UL_CI_84"].values[0], 3)) + "]")

    # Coupling metric
    booksheet_coupling = workbook.add_sheet("coupling", cell_overwrite_ok=True)
    coupling = ["ACAIC", "ACMIC", "AMMIC", "CBI", "CBO", "CountDeclMethodAll", "DACquote", "DMMEC", "ICP",
                "IHICP", "MPC", "NIHICP", "OCAEC", "OCMIC", "OMMEC", "OMMIC", "RFC"]
    for j in range(len(columns)):
        booksheet_coupling.write(0, j, columns[j])
    for i in range(len(coupling) + 1):
        if i == 0:
            continue
        print(i - 1, coupling[i - 1])
        booksheet_coupling.write(i, 0, coupling[i - 1])
        booksheet_coupling.write(i, 1, "[" + str(round(df[df["metric"] == coupling[i - 1]].loc[:, "LL_CI_84_adjusted"].values[0], 3)) + ",")
        booksheet_coupling.write(i, 2, str(round(df[df["metric"] == coupling[i - 1]].loc[:, "UL_CI_84_adjusted"].values[0], 3)) + "]")
        booksheet_coupling.write(i, 3, "[" + str(round(df[df["metric"] == coupling[i - 1]].loc[:, "LL_CI_84"].values[0], 3)) + ",")
        booksheet_coupling.write(i, 4, str(round(df[df["metric"] == coupling[i - 1]].loc[:, "UL_CI_84"].values[0], 3)) + "]")

    # Inheritance metric
    booksheet_inheritance = workbook.add_sheet("inheritance", cell_overwrite_ok=True)
    inheritance = ["CLD", "DPA", "DPD", "NMA", "NMI", "NMO", "NOC", "NOP", "PII", "SIX", "SP", "SPA", "ICH"]

    for j in range(len(columns)):
        booksheet_inheritance.write(0, j, columns[j])
    for i in range(len(inheritance) + 1):
        if i == 0:
            continue
        print(i - 1, inheritance[i - 1])
        # if inheritance[i - 1] == "PII":
        #     continue
        booksheet_inheritance.write(i, 0, inheritance[i - 1])
        booksheet_inheritance.write(i, 1, "[" + str(round(df[df["metric"] == inheritance[i - 1]].loc[:, "LL_CI_84_adjusted"].values[0], 3)) + ",")
        booksheet_inheritance.write(i, 2, str(round(df[df["metric"] == inheritance[i - 1]].loc[:, "UL_CI_84_adjusted"].values[0], 3)) + "]")
        booksheet_inheritance.write(i, 3, "[" + str(round(df[df["metric"] == inheritance[i - 1]].loc[:, "LL_CI_84"].values[0], 3)) + ",")
        booksheet_inheritance.write(i, 4, str(round(df[df["metric"] == inheritance[i - 1]].loc[:, "UL_CI_84"].values[0], 3)) + "]")

    workbook.save(directory + "doc_84CI_" + file.replace("csv", "xls"))

if __name__ == '__main__':
    s_time = time.time()
    m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled_all\\single_method\\supervised_meta_thresholds.csv"
    # m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled\\supervised\\AUC_supervised_metas.csv"
    # m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled_all\\single_method\\unsupervised_meta_thresholds.csv"
    # m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled\\unsupervised\\AUC_unsupervised_metas.csv"
    # m_dir_file = "F:\\NJU\\MTmeta\\experiments\\pooled_all\\Pooled_meta_thresholds.csv"

    doc_format(m_dir_file)
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of toDocFormat.py!\n",
          "The execution time of toDocFormat.py script is ", execution_time)
