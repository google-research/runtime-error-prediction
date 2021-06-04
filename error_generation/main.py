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

from misc_utils import get_codeforeces_paths
from get_trace import run_for_errors
from add_code import add_error

def main(base_path, trace_code_path, process_suffix="processed"):
	code_inp_data_paths = get_codeforeces_paths(base_path)
	for code_path, inp_paths in code_inp_data_paths:
		for idx, inp_path in enumerate(inp_paths):
			err_path = inp_path.replace(".txt", "_error.txt")
			out_path = inp_path.replace(".txt", "_out.txt")
			out_code_path = code_path.replace(".txt", "_"+str(idx)+"_perturbed.txt")
			data_trace_path = code_path.replace(".txt", "_"+str(idx)+"_trace.json")
			trace_successful = run_for_errors(code_path, data_trace_path, trace_code_path, inp_path, out_path, err_path, process_suffix)
			if trace_successful:
				data_trace_path = data_trace_path.replace(".json", "_"+process_suffix+".json")
				add_error(code_path, data_trace_path, "zero_err")


main("/Users/rishabgoel/Downloads/description2code_current/codeforces", "trace_code.py")