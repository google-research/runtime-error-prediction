from error_genration.misc_utils import (
    get_random_int,
    get_random_list_sample,
    get_random_float,
    get_random_int,
)

variable_names = [f"tmp{i}" for i in range(10)] + [chr(ord("a") + i) for i in range(26)]
num_range = [1, 100]


def get_zerro_expression_signature(var1, var2, val1, val2, val3, is_zerro_err):
    output_expr = [f"{var1}={val1}\n"]
    before_sub = get_random_int(0, 1)
    if before_sub:
        line = (
            f"{var2}={val2}/{var1}-{val1}\n"
            if is_zerro_err
            else f"{var2}={val2}/{var1}-{val3}\n"
        )
    else:
        line = (
            f"{var2}={val2}/{val1}-{var1}\n"
            if is_zerro_err
            else f"{var2}={val2}/{val3}-{var1}\n"
        )
    output_expr.append(line)
    return output_expr


def zerro_error_perturbation():
    sampled_vars = get_random_list_sample(variable_names, 2)
    samples_vals = get_random_float(*num_range, size=3)
    is_zerro_err = get_random_int(0, 1)
    return (
        get_zerro_expression_signature(*sampled_vars, *samples_vals, is_zerro_err),
        is_zerro_err,
    )
