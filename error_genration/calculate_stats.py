# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import redbaron as rb
import os
from collections import defaultdict
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import json
import math


def load_file(file_name):
    with open(file_name, "r") as file:
        code = file.read()
        return code


def get_codeforeces_paths(base_path):
    problem_paths = [
        os.path.join(base_path, problem_name_dir)
        for problem_name_dir in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, problem_name_dir))
    ]
    print(len(problem_paths))
    solution_paths = []
    for problem_path in problem_paths:
        solutions_path = os.path.join(problem_path, "solutions_python")
        if os.path.exists(solutions_path):
            solutions_path = [
                os.path.join(solutions_path, sol_name)
                for sol_name in os.listdir(solutions_path)
            ]
            solution_paths.append(solutions_path)

    solution_paths = [sol_path for path in solution_paths for sol_path in path]
    return solution_paths


def ceil(x):
    return int(math.ceil(x / 10.0)) * 10


def get_depth_stats_wrapper(files, block_types=["def", "class"]):
    stats_dict = {block_type: defaultdict(int) for block_type in block_types}
    stats_dict["len"] = defaultdict(int)
    err_files = []
    print(len(files))
    for file in files:
        code = load_file(file)
        code_ln = len(code.strip().split("\n"))
        stats_dict["len"][ceil(code_ln)] += 1
        is_err = get_depth_stats(code, stats_dict, block_types)
        if is_err:
            err_files.append(file)
    return stats_dict, err_files


def get_depth_stats(code, stats_dict, block_types):
    try:
        red = rb.RedBaron(code)
    except Exception as e:
        return True
    for block_type in block_types:
        curr_max = 0
        for block in red.find_all(block_type):
            depth = 0
            while block != red:
                depth += 1
                #                     while block.dumps()==block.parent.dumps():
                #                     import pdb;pdb.set_trace()
                block = block.parent
            curr_max = max(curr_max, depth)
        stats_dict[block_type][curr_max] += 1
    return False


def plot_data(x, y, xlabel, ylabel, name):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.bar(x, y)
    plt.savefig(name + ".pdf")
    # plt.show()


def save_csv(data, name):
    with open(name, "w") as file:
        file.write("\n".join(data))


def save_json(data, name):
    with open(name, "w") as file:
        json.dump(data, file)


def plot_wrapper(data):
    for key, value in data.items():
        x, y = zip(*value.items())
        plot_data(x, y, key, "Count", key)


def get_codechef_paths(base_path):
    pass


def main():
    paths = get_codeforeces_paths(
        "/home/mila/r/rishab.goel/description2code_current/codeforces"
    )
    codechef_paths = get_codechef_paths(
        "/home/mila/r/rishab.goel/description2code_current/codechef"
    )
    data, err_files = get_depth_stats_wrapper(paths)
    save_csv(err_files, "error.csv")
    save_csv(data, "calculated_stats.json")
    plot_wrapper(data)


main()
