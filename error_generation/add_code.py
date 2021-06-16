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
from misc_utils import (
    get_random_list_sample,
    load_json,
    load_data,
    write_csv,
    get_random_int,
    get_random_float,
)

VARIABLE_NAMES = ["tmp{}".format(i) for i in range(10)] + [
    chr(ord("a") + i) for i in range(26)
]
NUM_RANGE = [1, 100]


def get_perturb_line_step(code_trace):
    perturb_line = get_random_list_sample(code_trace.keys(), 1)[0]
    perturb_step = get_random_list_sample(code_trace[perturb_line], 1)[0]
    # import pdb;pdb.set_trace()
    perturb_var = get_random_list_sample(perturb_step.keys(), 1)[0]
    perturb_val = perturb_step[perturb_var]
    return int(perturb_line), perturb_var, int(perturb_val)


def get_zero_perturb_expression(perturb_var, perturb_val):
    assign_var = get_random_list_sample(VARIABLE_NAMES, 1)[0]
    is_zerro_err = get_random_int(0, 1)
    if is_zerro_err:
        numerator = get_random_float(*NUM_RANGE, size=1)[0]
        return (
            assign_var
            + "="
            + str(int(numerator))
            + "/"
            + str(perturb_val)
            + "-"
            + perturb_var,
            is_zerro_err,
        )
    else:
        perturb_val_offset, numerator = get_random_float(*NUM_RANGE, size=2)
        perturb_val = perturb_val + int(perturb_val_offset)
        return (
            assign_var
            + "="
            + str(int(numerator))
            + "/"
            + str(perturb_val)
            + "-"
            + perturb_var,
            is_zerro_err,
        )


def perturb_program(red, code_trace):
    perturb_line, perturb_var, perturb_val = get_perturb_line_step(code_trace)
    perturb_expression, is_err_present = get_zero_perturb_expression(
        perturb_var, perturb_val
    )
    red.at(perturb_line).insert_after(perturb_expression)
    return is_err_present


def add_error(org_code_fp, code_trace_fp, err_code_fp, suffx):
    code_trace = load_json(code_trace_fp)
    # To keep this function generic the name of the output
    # code file has the error type and indicator whether the
    # the error is present or not as suffix.
    err_code_fp = err_code_fp.replace(".txt", "_" + suffx + ".txt")
    program = load_data(org_code_fp).strip()
    red = rb.RedBaron(program)
    try:
        is_zerro_err = perturb_program(red, code_trace)
        err_code_fp = err_code_fp.replace(".txt", "_" + str(is_zerro_err) + ".txt")
    except Exception as e:
        # We can handle the exception as we want.
        # But for the time being we can return False.
        # import pdb;pdb.set_trace()
        return False

    write_csv(red.dumps(), err_code_fp)
    return True
