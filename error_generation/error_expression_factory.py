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


from misc_utils import (
    get_random_list_sample,
    load_json,
    load_data,
    write_csv,
    get_random_int,
    get_random_float,
)


class ErrorFactory:
    """TODO (rishab):
    1. implement methods for var name not defined and
       operand mismatch.
    2. make the expressions more complex.
    """

    VARIABLE_NAMES = ["tmp{}".format(i) for i in range(10)] + [
        chr(ord("a") + i) for i in range(26)
    ]
    NUM_RANGE = [1, 100]

    def __init__(self):
        self._builders = {
            "zero_err": self.get_zero_perturb_expression,
            "assert_err": self.get_assert_perturb_expression,
            "not_subscriptable_err": self.get_not_subscriptable_perturb_expression,
            "idx_out_range_err": self.get_index_range_perturb_expression,
            "undef_var_err": self.get_undef_name_perturb_expression,  # Caution: not implemented properly.
            "math_domain_err": self.get_math_domain_perturb_expression,
            "not_iterable_err": self.get_int_not_iterable_perturb_expression,
        }

    def get_zero_perturb_expression(self, perturb_var, perturb_val):
        assign_var = get_random_list_sample(self.VARIABLE_NAMES, 1)[0]
        is_zerro_err = get_random_int(0, 1)
        # is_zerro_err = 1
        if is_zerro_err:
            numerator = get_random_float(*self.NUM_RANGE, size=1)[0]
            return (
                assign_var
                + "="
                + str(int(numerator))
                + "/("
                + str(perturb_val)
                + "-"
                + perturb_var
                + ")",
                is_zerro_err,
            )
        else:
            perturb_val_offset, numerator = get_random_float(*self.NUM_RANGE, size=2)
            perturb_val = perturb_val + int(perturb_val_offset)
            return (
                assign_var
                + "="
                + str(int(numerator))
                + "/("
                + str(perturb_val)
                + "-"
                + perturb_var
                + ")",
                is_zerro_err,
            )

    def get_assert_perturb_expression(self, perturb_var, perturb_val):
        is_assert_err = get_random_int(0, 1)
        # is_assert_err = 1
        if is_assert_err:
            perturb_val_offset = get_random_float(*self.NUM_RANGE, size=1)[0]
            perturb_val = perturb_val + int(perturb_val_offset)
            return (
                "assert " + perturb_var + "==" + str(perturb_val),
                is_assert_err,
            )
        else:
            return (
                "assert " + perturb_var + "==" + str(perturb_val),
                is_assert_err,
            )

    def get_not_subscriptable_perturb_expression(self, perturb_var, perturb_val):
        is_not_subscriptable_err = get_random_int(0, 1)
        # is_not_subscriptable_err = 1
        if is_not_subscriptable_err:
            random_val, numerator = get_random_float(*self.NUM_RANGE, size=2)
            return (
                perturb_var + "[" + str(int(numerator)) + "] = " + str(int(random_val)),
                is_not_subscriptable_err,
            )
        else:
            return (
                "",
                is_not_subscriptable_err,
            )

    def get_index_range_perturb_expression(self, perturb_var, perturb_val):
        """This will occur very less frequently and hence we perhaps
        need to rethink how to handle generate the error.
        """
        is_index_range_err = get_random_int(0, 1)
        # is_index_range_err = 1
        if is_index_range_err:
            random_ass = get_random_float(*self.NUM_RANGE, size=1)[0]
            return (
                perturb_var
                + "["
                + str(len(perturb_val))
                + "] = "
                + str(int(random_ass)),
                is_index_range_err,
            )
        else:
            valid_idx = int(get_random_float(*[0, len(perturb_val) - 1], size=1)[0])
            random_ass = get_random_float(*self.NUM_RANGE, size=1)[0]
            return (
                perturb_var + "[" + str(valid_idx) + "] = " + str(random_ass),
                is_index_range_err,
            )

    def get_undef_name_perturb_expression(self, perturb_var, perturb_val):
        """Not implemented as per our requirements."""
        is_undef_name_err = get_random_int(0, 1)
        # is_undef_name_err = 1
        if is_undef_name_err:
            undef_var = get_random_list_sample(self.VARIABLE_NAMES, 1)[0]
            return (
                perturb_var + "=" + undef_var + "+" + str(perturb_val),
                is_undef_name_err,
            )
        else:
            return (
                "",
                is_undef_name_err,
            )

    def get_math_domain_perturb_expression(self, perturb_var, perturb_val):
        """The current implementation may cause unforeseen issues when the
        is_math_domain_err is 0 as the assign_var can be a part of the program. Also, we may
        perhaps need to refine how we import math module."""
        is_math_domain_err = get_random_int(0, 1)
        # is_math_domain_err = 1
        if is_math_domain_err:
            assign_var = get_random_list_sample(self.VARIABLE_NAMES, 1)[0]
            if perturb_val >= 0:
                random_ass = (
                    str(-1 * int(get_random_float(*self.NUM_RANGE, size=1)[0]))
                    + "*"
                    + perturb_var
                )
            else:
                random_ass = (
                    str(int(get_random_float(*self.NUM_RANGE, size=1)[0]))
                    + "*"
                    + perturb_var
                )
            return (
                "import math;"
                + assign_var
                + "="
                + "math.sqrt("
                + str(random_ass)
                + ")",
                is_math_domain_err,
            )
        else:
            assign_var = get_random_list_sample(self.VARIABLE_NAMES, 1)[0]
            if perturb_val >= 0:
                random_ass = (
                    str(int(get_random_float(*self.NUM_RANGE, size=1)[0]))
                    + "*"
                    + perturb_var
                )
            else:
                random_ass = (
                    str(-1 * int(get_random_float(*self.NUM_RANGE, size=1)[0]))
                    + "*"
                    + perturb_var
                )
            return (
                "import math;"
                + assign_var
                + "="
                + "math.sqrt("
                + str(random_ass)
                + ")",
                is_math_domain_err,
            )

    def _relevant_operand_val_type(self, val, is_same):
        pass

    def get_operand_type_mismatch_perturb_expression(self, perturb_var, perturb_val):
        pass

    def get_int_not_iterable_perturb_expression(self, perturb_var, perturb_val):
        """TODO: 1. Add more variants of the for loop.
        2. Add logic to include the while loop.
        """
        is_int_not_iterable_err = get_random_int(0, 1)
        # is_int_not_iterable_err = 1
        if is_int_not_iterable_err:
            assign_var = get_random_list_sample(self.VARIABLE_NAMES, 1)[0]
            random_ass = int(get_random_float(*self.NUM_RANGE, size=1)[0])
            return (
                "{}=[{}+val for val in {}]".format(assign_var, random_ass, perturb_var),
                is_int_not_iterable_err,
            )
        else:
            return "", is_int_not_iterable_err

    def add_err(self, err_type, perturb_var, perturb_val):
        expr_builder = self._builders.get(err_type.lower(), None)
        if not expr_builder:
            raise ValueError(err_type + " is not a valid error generation function.")
        return expr_builder(perturb_var, perturb_val)
