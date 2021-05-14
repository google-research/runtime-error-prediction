import os
import random
import copy
import redbaron as rb
import numpy as np


import concurrent.futures as cf

from error_genration.misc_utils import (
    get_random_int,
    load_dataa,
    get_codeforeces_paths,
    write_csv,
)
from error_genration.err_expr_utils import zerro_error_perturbation


def add_perturbation(perturb_node, expr_ass, expr_ass_line, expr_err, expr_err_line):
    # import pdb;pdb.set_trace()
    print("yo")
    perturb_node.at(expr_ass_line).insert_before(expr_ass)
    print("yo1")
    perturb_node.at(expr_err_line).insert_before(expr_err)


def get_ass_expr_lines(code_lines, apart=0.75):
    ln = len(code_lines)
    assert ln > 0
    assign_upper_range = max(1, int((1 - apart) * ln))
    if assign_upper_range == 1:
        ass_line = 1
    else:
        ass_line = get_random_int(1, assign_upper_range)

    for counter in range(20):
        if (
            to_include(code_lines[ass_line - 1])
            or ass_line == ln
            or assign_upper_range == 1
        ):
            break
        ass_line = get_random_int(1, int((1 - apart) * ln))
    if not to_include(code_lines[ass_line - 1]):
        return -1, -1

    expr_line = min(ln, ass_line + int(apart * ln))
    for counter in range(20):
        if to_include(code_lines[expr_line - 1]) or expr_line == ln:
            break
        expr_line = get_random_int(min(ln, ass_line + int(apart * ln)), ln)
    if not to_include(code_lines[expr_line - 1]):
        return -1, -1
    return ass_line, expr_line


def get_parent_node(line_to_perturb, red):
    selected_node = red.at(line_to_perturb)
    print(selected_node.dumps())
    if "if __name__ == '__main__':" in selected_node.parent.dumps():
        if len(selected_node.dumps().split("\n")) < 5:
            return None, False
        else:
            return selected_node.parent, True
    else:
        out_node = selected_node
        while out_node != red and "def" not in out_node.dumps():
            if out_node.dumps() == out_node.parent.dumps():
                break
            out_node = out_node.parent
        return out_node, True


def to_include(line):
    for token in ["if", "while", "for", "def", "class", "import", "else"]:
        if token in line:
            return False
    if not line.strip():
        return False
    line_rb = rb.RedBaron(line.strip())
    if isinstance(line_rb[0], rb.nodes.CommentNode) or isinstance(
        line_rb[0], rb.nodes.EndlNode
    ):
        return False
    return True


def get_perturb_node(red, program_source, program_ln):
    perturb_node = None
    for counter in range(20):
        line_to_perturb = get_random_int(1, program_ln)
        if to_include(program_source[line_to_perturb - 1]):
            break
    if to_include(program_source[line_to_perturb - 1]):
        perturb_node, found_correct_location = get_parent_node(line_to_perturb, red)
    if not perturb_node:
        perturb_node = red
    return perturb_node


def perturb_program(program_fp, suffx="perturbed"):
    output_fp = program_fp.replace(".txt", f"_{suffx}.txt")

    program = load_dataa(program_fp).strip()

    try:
        red = rb.RedBaron(program)
        program_lines = program.split("\n")
        program_ln = len(program_lines)
        perturb_node = get_perturb_node(red, program_lines, program_ln)

        [expr_ass, expr_err], is_err = zerro_error_perturbation()
        perturb_node_lines = perturb_node.dumps().split("\n")
        expr_ass_line, expr_err_line = get_ass_expr_lines(
            perturb_node_lines, apart=0.75
        )

        if (
            not perturb_node_lines[expr_ass_line - 1]
            or not perturb_node_lines[expr_err_line - 1]
            or expr_ass_line == -1
        ):
            return "", -1, "Not found a good line"
        add_perturbation(perturb_node, expr_ass, expr_ass_line, expr_err, expr_err_line)
        # write_csv(program.dumps(), output_fp)
    except Exception as e:
        return "", -1, f"{e}"
    return output_fp, is_err, None


def perturb_program_wrapper(paths):

    output_paths = []
    errors = []
    for path in paths:
        print(path)
        if path.endswith(".txt") or path.endswith(".py"):
            out_path, label, error = perturb_program(path, suffx="perturbed")
            if error:
                errors.append(f"{path}:\n {error}")
            else:
                output_paths.append(f"{out_path},{label}")
        else:
            errors.append(f"{path}:\n format error")
    return (output_paths, errors)


def concurrent_program_perturbation(paths, num_processes, out_path="./"):
    per_process_paths = np.array_split(paths, num_processes)
    output_paths, errors = [], []
    with cf.ProcessPoolExecutor() as executor:
        results = [
            executor.submit(perturb_program_wrapper, per_process_paths[process_num])
            for process_num in range(num_processes)
        ]
        for completed in cf.as_completed(results):
            res_out_paths, res_errs = completed.result()
            output_paths.extend(res_out_paths)
            errors.append(res_errs)
    # write_csv("\n".join(output_paths), f"{out_path}/label_file.csv")
    # write_csv("\n".join(errors), f"{out_path}/error_file.csv")


def main():
    code_forces_paths = get_codeforeces_paths(
        "/home/mila/r/rishab.goel/description2code_current/codeforces"
    )
    # concurrent_program_perturbation(code_forces_paths[:4], 3)
    res_out_paths, res_errs = perturb_program_wrapper(code_forces_paths[:100])
    # import pdb;pdb.set_trace()


if __name__ == "__main__":
    # data = load_dataa("/home/mila/r/rishab.goel/description2code_current/codechef/easy/ACBALL/solutions_python/10211792.txt")
    main()
