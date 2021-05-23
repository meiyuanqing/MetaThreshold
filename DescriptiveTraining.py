#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/5/23
Time: 11:55
File: DescriptiveTraining.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Descriptive statistics for the training data sets and test data sets
i.e., Number of classes and Number of bugs of versions of each project: Max. Min. Mean Std.

"""
import time


def Descriptive_data(t_dir="F:\\NJU\\MTmeta\\experiments\\trainings\\",
                     d_dir="F:\\NJU\\MTmeta\\experiments\\descriptive\\",
                     training_list="List.txt",
                     project_list="project_List.txt"):
    import os
    import csv
    import numpy as np
    import pandas as pd

    os.chdir(d_dir)
    print(os.getcwd())

    with open(t_dir + training_list) as l:
        lines = l.readlines()

    with open(t_dir + project_list) as l_project:
        project = l_project.readlines()

    print(project)
    for i in range(len(project)):
        project[i] = project[i].replace("\n", "")
    print(project)
    class_df = pd.DataFrame(columns=project)
    bug_df = pd.DataFrame(columns=project)
    print(class_df)
    print(bug_df)

    class_total = []
    bug_total = []

    for line in lines:
        file = line.replace("\n", "")
        print('the file is ', file)

        df = pd.read_csv(t_dir + file)
        Number_of_classes = len(df)
        Number_of_bugs = df["bug"].sum()

        project_file = file.split("-")[0]

        print("\tNumber_of_classes's len is ", Number_of_classes,
              "\tNumber_of_bugs's len is ", Number_of_bugs,
              "\tproject_file is ", project_file)

        class_df.loc[len(class_df[project_file].dropna()), project_file] = Number_of_classes
        bug_df.loc[len(bug_df[project_file].dropna()), project_file] = Number_of_bugs
        class_total.append(Number_of_classes)
        bug_total.append(Number_of_bugs)
        print(class_df)
        print(bug_df)

    with open(d_dir + "descriptive_training.csv", 'a+', encoding="utf-8", newline='') as f:
        writer_f = csv.writer(f)
        if os.path.getsize(d_dir + "descriptive_training.csv") == 0:
            writer_f.writerow(["project_names", "Number_of_classes_max", "Number_of_classes_min",
                               "Number_of_classes_mean", "Number_of_classes_std", "Number_of_bugs_max",
                               "Number_of_bugs_min", "Number_of_bugs_mean", "Number_of_bugs_std"])
            # writer_f.writerow(["project_names", "Number_of_classes", "Number_of_bugs"])
        for project_i in project:
            max_class = class_df[project_i].max()
            min_class = class_df[project_i].min()
            mean_class = class_df[project_i].mean()
            std_class = class_df[project_i].std()
            max_bug = bug_df[project_i].max()
            min_bug = bug_df[project_i].min()
            mean_bug = bug_df[project_i].mean()
            std_bug = bug_df[project_i].std()
            writer_f.writerow([project_i, max_class, min_class, mean_class, std_class,
                               max_bug, min_bug, mean_bug, std_bug])
        writer_f.writerow([project_i, np.max(class_total), np.min(class_total), np.mean(class_total),
                           np.std(class_total), np.max(bug_total), np.min(bug_total), np.mean(bug_total),
                           np.std(bug_total)])


if __name__ == '__main__':
    s_time = time.time()

    # Descriptive_data()
    Descriptive_data(t_dir="F:\\NJU\\MTmeta\\experiments\\testings\\",
                     d_dir="F:\\NJU\\MTmeta\\experiments\\descriptive\\testings\\",
                     training_list="List.txt",
                     project_list="project_List.txt")

    e_time = time.time()
    execution_time = e_time - s_time
    print("The __name__ is ", __name__, ".\tThis is end of DescriptiveTraining.py!",
          "\tThe execution time is ", execution_time)
