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

import random
import numpy as np
import os
import json
import yaml
import copy
from collections import defaultdict

ALLOWED_TYPE = {
    "zero_err": [int, float],
    "assert_err": [int, float],
    "not_subscriptable_err": [int, float],
    "idx_out_range_err": [list],
    "undef_var_err": [int, float, str, list],
    "math_domain_err": [int, float],
    "not_iterable_err": [int, float],
}


def get_codeforces_inp_data_paths(base_path):
    inp_data_directory = os.path.join(base_path, "samples")
    inp_data_paths = [
        inp_data_directory + "/" + path
        for path in os.listdir(inp_data_directory)
        if "input" in path
    ]
    return inp_data_paths


def create_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_codeforeces_paths(base_path):
    problem_paths = [
        os.path.join(base_path, problem_name_dir)
        for problem_name_dir in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, problem_name_dir))
    ]
    # print(len(problem_paths))
    data_paths = []
    for problem_path in problem_paths:
        prob_solutions_path = os.path.join(problem_path, "solutions_python")
        # print prob_solutions_path
        perturbed_prob_solutions_path = os.path.join(
            problem_path, "perturbed_solutions_python"
        )
        create_dirs(perturbed_prob_solutions_path)
        trace_path = os.path.join(problem_path, "trace")
        create_dirs(trace_path)
        err_out_path = os.path.join(problem_path, "err_out")
        create_dirs(err_out_path)
        inp_data_paths = get_codeforces_inp_data_paths(problem_path)
        # import pdb;pdb.set_trace()
        if os.path.exists(prob_solutions_path):
            solution_paths = []
            for sol_name in os.listdir(prob_solutions_path):
                code_path = os.path.join(prob_solutions_path, sol_name)
                sol_name_json = sol_name.replace(".txt", ".json")
                perturbed_code_path = os.path.join(
                    perturbed_prob_solutions_path, sol_name
                )
                trace_code_path = os.path.join(trace_path, sol_name_json)
                sol_err_out_path = os.path.join(err_out_path, sol_name)
                solution_paths.append(
                    (
                        code_path,
                        inp_data_paths,
                        perturbed_code_path,
                        trace_code_path,
                        sol_err_out_path,
                    )
                )
            data_paths.append(solution_paths)

    data_paths = [sol_path for path in data_paths for sol_path in path]
    return data_paths


def is_valid_type(val, type):
    return isinstance(val, type)


def get_valid_code_trace(code_trace, err_suffx):
    code_trace_filtered = defaultdict(list)
    for line in code_trace:
        for step_idx in range(len(code_trace[line])):
            new_step_dict = {}
            for var in code_trace[line][step_idx]:
                if any(
                    is_valid_type(code_trace[line][step_idx][var], typ)
                    for typ in ALLOWED_TYPE[err_suffx]
                ):
                    new_step_dict[var] = code_trace[line][step_idx][var]
            if new_step_dict:
                code_trace_filtered[line].append(new_step_dict)
    return code_trace_filtered


def set_seeds(seed=10):
    random.seed(seed)
    np.random.seed(seed)


def load_data(fp):
    with open(fp, "r") as file:
        data = file.read().strip()
    return data


def load_json(fp):
    with open(fp, "r") as file:
        return json.load(file)


def load_yaml(fp):
    with open(fp, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def write_csv(data, fp):
    with open(fp, "w") as file:
        file.write(data)


def get_random_int(lower_limit, upper_limit):
    return random.randint(lower_limit, upper_limit)


def get_random_float(lower_limit, upper_limit, size=None):
    return np.random.uniform(lower_limit, upper_limit, size=size)


def get_random_list_sample(lst, num_samples):
    return random.sample(lst, num_samples)
