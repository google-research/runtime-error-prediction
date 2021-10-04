import fire

NO_DATA = 'No data'
TIMEOUT = 'Timeout'

ERROR_KINDS = [
    'AssertionError',  # index = 4
    'AttributeError',
    # bdb.BdbQuit p02702 s706694213
    'decimal.InvalidOperation',
    'EOFError',
    'FileNotFoundError',
    'ImportError',
    'IndentationError',  # index = 10
    'IndexError',
    'KeyError',
    'MathDomainError',
    'MemoryError',
    'ModuleNotFoundError',
    'NameError',  # index = 16
    'OSError',  # Bad file descriptor
    'OverflowError',
    're.error',  # nothing to repeat at position 0
    'RecursionError',
    'RuntimeError',
    'StopIteration',
    'SyntaxError',
    'TabError',
    'TypeError',  # index = 25
    'UnboundLocalError',
    'ValueError',  # index = 27
    'ZeroDivisionError',
    'numpy.AxisError',  # index = 29
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

ALL_ERROR_KINDS = [NO_DATA, NO_ERROR, SILENT_ERROR, OTHER_ERROR, TIMEOUT] + ERROR_KINDS
NUM_CLASSES = len(ALL_ERROR_KINDS)


def to_index(error_kind):
  if error_kind == NO_DATA:
    return 0
  if error_kind in [NO_ERROR, NO_ERROR_WITH_STDERR]:
    return 1
  if error_kind in [SILENT_ERROR]:
    return 2
  if error_kind.startswith(OTHER_ERROR):
    return 3
  elif error_kind == TIMEOUT:
    return 4
  if error_kind in ERROR_KINDS:
    return 5 + ERROR_KINDS.index(error_kind)
  return 3  # Other.


def to_error(index):
  return ALL_ERROR_KINDS[index]


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
NO_DATA_ID = to_index(NO_DATA)
NO_ERROR_ID = to_index(NO_ERROR)

if __name__ == '__main__':
  fire.Fire()
