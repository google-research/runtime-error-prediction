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
from misc_utils import get_random_list_sample, load_json, load_data, write_csv

def get_perturb_line_step(code_trace):
	perturb_line = get_random_list_sample(code_trace.keys(), 1)[0]
	perturb_step = get_random_list_sample(code_trace[perturb_line],1)[0]
	# import pdb;pdb.set_trace()
	perturb_var = get_random_list_sample(perturb_step.keys(),1)[0]
	perturb_val = perturb_step[perturb_var]
	return int(perturb_line), perturb_var, int(perturb_val)

def get_perturb_expression(perturb_var, perturb_val):
	return 'tmp1 = 1/'+str(perturb_val)+ '-'+ perturb_var, True

def perturb_program(red, code_trace):
	perturb_line, perturb_var, perturb_val = get_perturb_line_step(code_trace)
	perturb_expression, is_err_present = get_perturb_expression(perturb_var, perturb_val)
	# print perturb_expression, perturb_line
	# import pdb;pdb.set_trace()
	red.at(perturb_line).insert_after(perturb_expression)

def add_error(org_code_fp, code_trace_fp, suffx):
	code_trace = load_json(code_trace_fp)
	err_code_fp = org_code_fp.replace(".txt", "_"+suffx+".txt")
	program = load_data(org_code_fp).strip()
	red = rb.RedBaron(program)
	_ = perturb_program(red, code_trace)
	write_csv(red.dumps(), err_code_fp)