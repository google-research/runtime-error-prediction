NO_DATA = 'No data'
TIMEOUT = 'Timeout'

ERROR_KINDS = [
    'AssertionError',
    'AttributeError',
    # bdb.BdbQuit p02702 s706694213
    'decimal.InvalidOperation',
    'EOFError',
    'FileNotFoundError',
    'ImportError',
    'IndentationError',
    'IndexError',
    'KeyError',
    'MathDomainError',
    'MemoryError',
    'ModuleNotFoundError',
    'NameError',
    'OSError', # Bad file descriptor
    'OverflowError',
    're.error',  # nothing to repeat at position 0
    'RecursionError',
    'RuntimeError',
    'StopIteration',
    'SyntaxError',
    'TabError',
    'TypeError',
    'UnboundLocalError',
    'ValueError',
    'ZeroDivisionError',
    'numpy.AxisError',
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


if __name__ == '__main__':
  fire.Fire()
