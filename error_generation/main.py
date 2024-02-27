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

import os

from error_generation.misc_utils import get_codeforeces_paths, load_yaml, set_seeds
from error_generation.get_trace import run_for_errors
from error_generation.add_code import add_error
from error_generation.error_expression_factory import ErrorFactory

"""
TODO(rishab): Setup code to include codechef as well.

Run instructions:
In the compressive-ipagnn folder run the following command:
python -m error_generation.main
"""


def main(config_fp):
    config = load_yaml(config_fp)
    set_seeds()
    code_inp_data_paths = get_codeforeces_paths(config["base_path"])
    error_expr_factory_obj = ErrorFactory()
    for (
        code_path,
        inp_paths,
        perturbed_code_path,
        trace_data_path,
        sol_err_out_path,
    ) in code_inp_data_paths:
        for idx, inp_path in enumerate(inp_paths):
            # print inp_path
            err_path = sol_err_out_path.replace(
                ".txt", "_error" + "_" + str(idx) + ".txt"
            )
            out_path = sol_err_out_path.replace(
                ".txt", "_out" + "_" + str(idx) + ".txt"
            )
            out_code_path = perturbed_code_path.replace(".txt", "_" + str(idx) + ".txt")
            data_trace_path = trace_data_path.replace(
                ".json", "_trace_" + str(idx) + ".json"
            )
            is_trace_successful = run_for_errors(
                code_path,
                data_trace_path,
                config["trace_code_path"],
                inp_path,
                out_path,
                err_path,
                config["process_suffix"],
            )
            if is_trace_successful:
                data_trace_path = data_trace_path.replace(
                    ".json", "_" + config["process_suffix"] + ".json"
                )
                for err_suffix in config["errors"]:
                    # import pdb;pdb.set_trace()
                    _ = add_error(
                        code_path,
                        data_trace_path,
                        out_code_path,
                        err_suffix,
                        error_expr_factory_obj,
                    )
            # break
        # break


if __name__ == "__main__":
    main("error_generation/config.yaml")
