import pytest
import subprocess
from error_generation.error_expression_factory import ErrorFactory

class TestErrorFactory():
    """The test suite is not complete but will test some obvious
       issues in code."""
    def setup(self):
        self.error_factory = ErrorFactory()
        self.test_var_name = "test_var"
        self.test_var_val = 23
        self.test_lst_var_name = "test_lst_var"
        self.test_lst_var_val = [1,2,3,4]
        self.test_var_assign = self.test_var_name + "=" + str(self.test_var_val) + "\n"
        self.test_lst_var_assign = self.test_lst_var_name + "=" + str(self.test_lst_var_val) + "\n"

    def test_get_zero_perturb_expression(self):
        
        expr, is_err_present = self.error_factory.add_err("zero_err", self.test_var_name, self.test_var_val)
        expr = self.test_var_assign + expr
        
        while not is_err_present:
            subprocess_call = subprocess.call(
                ["python", "-c", expr], stderr=subprocess.PIPE
            )
            assert subprocess_call == 0
            expr, is_err_present = self.error_factory.add_err("zero_err", self.test_var_name, self.test_var_val)
            expr = self.test_var_assign + expr

        with pytest.raises(subprocess.CalledProcessError) as exc:
            try:
                subprocess_call = subprocess.check_output(
                    ["python", "-c", expr], stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as exception:
                if "ZeroDivisionError" in exception.output:
                    raise exception
    
    def test_get_assert_perturb_expression(self):
        expr, is_err_present = self.error_factory.add_err("assert_err", self.test_var_name, self.test_var_val)
        expr = self.test_var_assign + expr
        while not is_err_present:
            subprocess_call = subprocess.call(
                ["python", "-c", expr], stderr=subprocess.PIPE
            )
            assert subprocess_call == 0
            expr, is_err_present = self.error_factory.add_err("assert_err", self.test_var_name, self.test_var_val)
            expr = self.test_var_assign + expr
        with pytest.raises(subprocess.CalledProcessError) as exc:
            try:
                subprocess_call = subprocess.check_output(
                    ["python", "-c", expr], stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as exception:
                if "AssertionError" in exception.output:
                    raise exception
    
    def test_get_not_subscriptable_perturb_expression(self):
        expr, is_err_present = self.error_factory.add_err("not_subscriptable_err", self.test_var_name, self.test_var_val)
        expr = self.test_var_assign + expr
        while not is_err_present:
            subprocess_call = subprocess.call(
                ["python", "-c", expr], stderr=subprocess.PIPE
            )
            assert subprocess_call == 0
            expr, is_err_present = self.error_factory.add_err("not_subscriptable_err", self.test_var_name, self.test_var_val)
            expr = self.test_var_assign + expr
        with pytest.raises(subprocess.CalledProcessError) as exc:
            try:
                subprocess_call = subprocess.check_output(
                    ["python", "-c", expr], stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as exception:
                if "TypeError" in exception.output and "not support item assignment" in exception.output:
                    raise exception
    
    def test_get_index_range_perturb_expression(self):
        expr, is_err_present = self.error_factory.add_err("idx_out_range_err", self.test_lst_var_name, self.test_lst_var_val)
        expr = self.test_lst_var_assign + expr

        while not is_err_present:
            subprocess_call = subprocess.call(
                ["python", "-c", expr], stderr=subprocess.PIPE
            )
            if subprocess_call!=0:
                import pdb;pdb.set_trace()
            assert subprocess_call == 0

            expr, is_err_present = self.error_factory.add_err("idx_out_range_err", self.test_lst_var_name, self.test_lst_var_val)
            expr = self.test_lst_var_assign + expr

        with pytest.raises(subprocess.CalledProcessError) as exc:
            try:
                subprocess_call = subprocess.check_output(
                    ["python", "-c", expr], stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as exception:
                if "IndexError" in exception.output and "index out of range" in exception.output:
                    raise exception
    
    def test_get_math_domain_perturb_expression(self):
        expr, is_err_present = self.error_factory.add_err("math_domain_err", self.test_var_name, self.test_var_val)
        expr = self.test_var_assign + expr

        while not is_err_present:
            subprocess_call = subprocess.call(
                ["python", "-c", expr], stderr=subprocess.PIPE
            )
            assert subprocess_call == 0
            expr, is_err_present = self.error_factory.add_err("math_domain_err", self.test_var_name, self.test_var_val)
            expr = self.test_var_assign + expr
        
        with pytest.raises(subprocess.CalledProcessError) as exc:
            try:
                subprocess_call = subprocess.check_output(
                    ["python", "-c", expr], stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as exception:
                if "ValueError" in exception.output and "math domain error" in exception.output:
                    raise exception
    
    def test_get_int_not_iterable_perturb_expression(self):
        expr, is_err_present = self.error_factory.add_err("not_iterable_err", self.test_var_name, self.test_var_val)
        expr = self.test_var_assign + expr

        while not is_err_present:
            subprocess_call = subprocess.call(
                ["python", "-c", expr], stderr=subprocess.PIPE
            )
            assert subprocess_call == 0
            expr, is_err_present = self.error_factory.add_err("not_iterable_err", self.test_var_name, self.test_var_val)
            expr = self.test_var_assign + expr

        with pytest.raises(subprocess.CalledProcessError) as exc:
            try:
                subprocess_call = subprocess.check_output(
                    ["python", "-c", expr], stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as exception:
                if "TypeError" in exception.output and "object is not iterable" in exception.output:
                    raise exception

    def test_get_operand_type_mismatch_perturb_expression(self):
        pass

    def test_get_undef_name_perturb_expression(self):
        pass