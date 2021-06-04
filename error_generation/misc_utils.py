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

def get_codeforces_inp_data_paths(base_path):
    inp_data_directory = os.path.join(base_path, "samples")
    inp_data_paths = [inp_data_directory+"/"+path for path in os.listdir(inp_data_directory) if "input" in path]
    return inp_data_paths


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
        inp_data_paths = get_codeforces_inp_data_paths(problem_path)
        # import pdb;pdb.set_trace()
        if os.path.exists(solutions_path):
            solutions_path = [
                (os.path.join(solutions_path, sol_name), inp_data_paths)
                for sol_name in os.listdir(solutions_path)
            ]
            solution_paths.append(solutions_path)

    solution_paths = [sol_path for path in solution_paths for sol_path in path]
    return solution_paths


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

def write_csv(data, fp):
    with open(fp, "w") as file:
        file.write(data)


def get_random_int(lower_limit, upper_limit):
    return random.randint(lower_limit, upper_limit)


def get_random_float(lower_limit, upper_limit, size=None):
    return np.random.uniform(lower_limit, upper_limit, size=size)


def get_random_list_sample(lst, num_samples):
    return random.sample(lst, num_samples)