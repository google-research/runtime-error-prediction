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
    get_valid_code_trace,
)

VARIABLE_NAMES = ["tmp{}".format(i) for i in range(10)] + [
    chr(ord("a") + i) for i in range(26)
]
NUM_RANGE = [1, 100]


def get_perturb_line_step(code_trace_org, err_suffx):
    # Not all the variable types are valid for all the errors.
    # For instance for index out of range an int var is not valid.
    code_trace = get_valid_code_trace(code_trace_org, err_suffx)
    if not code_trace:
        return None, None, None
    perturb_line = get_random_list_sample(code_trace.keys(), 1)[0]
    perturb_step = get_random_list_sample(code_trace[perturb_line], 1)[0]
    perturb_var = get_random_list_sample(perturb_step.keys(), 1)[0]
    perturb_val = perturb_step[perturb_var]
    return int(perturb_line), perturb_var, perturb_val


def perturb_program(red, code_trace, err_suffx, error_expr_factory_obj):
    perturb_line, perturb_var, perturb_val = get_perturb_line_step(
        code_trace, err_suffx
    )
    if perturb_line is None:
        return 0
    perturb_expression, is_err_present = error_expr_factory_obj.add_err(
        err_suffx, perturb_var, perturb_val
    )
    # TODO(rishab): Need to be careful to ensure that that the insertion
    # line is not an AssignmentNode in RedBaron.
    if err_suffx == "math_domain_err":
        # The sqrt function needs to be imported so that sqrt function
        # can be called. I am not sure if we can just add the expression
        # without proper imports.
        import_statement, perturb_expression = perturb_expression.split(";")
        red.at(perturb_line).insert_before(import_statement, offset=perturb_line - 1)
        red.at(perturb_line + 1).insert_after(perturb_expression)
    else:
        red.at(perturb_line).insert_after(perturb_expression)
    return is_err_present


def add_error(
    org_code_fp, code_trace_fp, err_code_fp, err_suffx, error_expr_factory_obj
):
    # We can optimize the code by passing the read file.
    # But for now to ensure isolation, I am doing it
    # explicitly.
    code_trace = load_json(code_trace_fp)
    # To keep this function generic the name of the output
    # code file has the error type and indicator whether the
    # the error is present or not as suffix.
    err_code_fp = err_code_fp.replace(".txt", "-" + err_suffx + ".txt")
    program = load_data(org_code_fp).strip()
    red = rb.RedBaron(program)
    try:
        is_err_present = perturb_program(
            red, code_trace, err_suffx, error_expr_factory_obj
        )
        err_code_fp = err_code_fp.replace(".txt", "-" + str(is_err_present) + ".txt")
    except Exception as e:
        # We can handle the exception as we want.
        # But for the time being we can return False.
        # import pdb;pdb.set_trace()
        return False

    write_csv(red.dumps(), err_code_fp)
    return True
