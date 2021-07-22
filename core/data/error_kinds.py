<<<<<<< HEAD
import fire

=======
>>>>>>> 0285ec2... Reorganize error kinds
NO_DATA = 'No data'
TIMEOUT = 'Timeout'

ERROR_KINDS = [
<<<<<<< HEAD
    'AssertionError',  # index = 4
=======
    'AssertionError',
>>>>>>> 0285ec2... Reorganize error kinds
    'AttributeError',
    # bdb.BdbQuit p02702 s706694213
    'decimal.InvalidOperation',
    'EOFError',
    'FileNotFoundError',
    'ImportError',
<<<<<<< HEAD
    'IndentationError',  # index = 10
=======
    'IndentationError',
>>>>>>> 0285ec2... Reorganize error kinds
    'IndexError',
    'KeyError',
    'MathDomainError',
    'MemoryError',
    'ModuleNotFoundError',
<<<<<<< HEAD
    'NameError',  # index = 16
=======
    'NameError',
>>>>>>> 0285ec2... Reorganize error kinds
    'OSError', # Bad file descriptor
    'OverflowError',
    're.error',  # nothing to repeat at position 0
    'RecursionError',
    'RuntimeError',
    'StopIteration',
    'SyntaxError',
    'TabError',
<<<<<<< HEAD
    'TypeError',  # index = 25
    'UnboundLocalError',
    'ValueError',
    'ZeroDivisionError',
    'numpy.AxisError',  # index = 29
=======
    'TypeError',
    'UnboundLocalError',
    'ValueError',
    'ZeroDivisionError',
    'numpy.AxisError',
>>>>>>> 0285ec2... Reorganize error kinds
    # 'Exception',
    # SyntaxError: invalid syntax
    # SyntaxError: invalid character
    # SyntaxError: import * only allowed at module level
    # SyntaxError: closing parenthesis
    # SyntaxError: cannot assign to operator
    # SyntaxError: Missing parentheses in call to
    # SyntaxError: from __future__ imports must occur at the beginning of the file
    # SyntaxError: invalid non-printable character
]

OTHER_ERROR = 'Other'
SILENT_ERROR = 'Silent error. Nothing in stderr.'
NO_ERROR_WITH_STDERR = 'No error (but using stderr anyway)'
NO_ERROR = 'No error'
<<<<<<< HEAD

NUM_CLASSES = 3 + len(ERROR_KINDS)

def to_index(error_kind):
  if error_kind == NO_DATA:
    return 0
  if error_kind in [NO_ERROR, NO_ERROR_WITH_STDERR]:
    return 1
  if error_kind in [SILENT_ERROR]:
    return 2
  if error_kind.startswith(OTHER_ERROR):
    return 3
  if error_kind in ERROR_KINDS:
    return 4 + ERROR_KINDS.index(error_kind)
  return 3  # Other.


def to_error(index):
  error_kinds_list = [NO_DATA, NO_ERROR, SILENT_ERROR, OTHER_ERROR] + ERROR_KINDS
  return error_kinds_list[index]


# "Tier 1" errors are those that are definitively
# execution-behavior-based runtime errors.
TIER1_ERROR_KINDS = [
    'IndexError',
    'TypeError',
    TIMEOUT,
    'ValueError',
    'AttributeError',
    'KeyError',
    'ZeroDivisionError',
    'AssertionError',
    'MathDomainError',
    'NameError',
]
TIER1_ERROR_IDS = [to_index(e) for e in TIER1_ERROR_KINDS]


if __name__ == '__main__':
  fire.Fire()
=======
>>>>>>> 0285ec2... Reorganize error kinds
