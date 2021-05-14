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


def set_seeds(seed=10):
    random.seed(seed)
    np.random.seed(seed)


def load_dataa(fp):
    with open(fp, "r", encoding="utf-8") as file:
        data = file.read().strip()
    return data


def write_csv(data, fp):
    with open(fp, "w") as file:
        file.write(data)


def get_random_int(lower_limit, upper_limit):
    return random.randint(lower_limit, upper_limit)


def get_random_float(lower_limit, upper_limit, size=None):
    return np.random.uniform(lower_limit, upper_limit, size=size)


def get_random_list_sample(lst, num_samples):
    return random.sample(lst, num_samples)
