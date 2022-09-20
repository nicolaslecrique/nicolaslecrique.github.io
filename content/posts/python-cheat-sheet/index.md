---
title: 'Python cheatsheets for machine learning'
date: 2022-09-18T10:48:07+02:00
draft: false
math: true
images: []
description: null
resources:
- name: "featured-image"
  src: "featured-image.jpg"
---

A list of my Python cheatsheets for machine learning and more

<!--more-->

## Native Python


```python
from datetime import datetime
from math import log, sqrt, sin, pi, e, factorial

# useful commands
"""
python --version
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
"""

# Containers, types
li: list[int] = [1, 2]
tu: tuple[int, int] = (1, 2)
se: set[int] = {1, 2}
di: dict[str, int] = {"a": 1, "b": 2}

# multiple assignment, unpacking and unpacking operator *
start, stop, step = 1, 6, 2
one, two, three = range(3)  # unpacking works with tuples and iterable types
*six_to_height, nine = [6, 7, 8, 9]  # six_to_height == [6, 7, 8]

# list comprehension (replace map->filter pattern), range
list_comp = [even for even in range(start, stop, step) if even != 3]  # [1, 5]
dict_comp = {str(i): i for i in range(stop)}  # {"0": 0, "1": 1, ..., "5": 5}

# list indexing and operations
l = [0, 1, 2, 3, 4, 5, 6]
last, before_last = l[-1], l[-2]  # 6, 5
slice_step = l[start:stop:step]  # [1, 3, 5]
slice_defaults_ok = l[::] == l and l[:] == l and l[0:len(l):1] == l
l_min, l_len, l_sort, all_true, any_true = min(l), len(l), sorted(l), all(l), any(l)
zipped_l_and_rev: list[tuple[int, int]] = list(zip(l, reversed(l)))  # [(0,6), (1,5)...(6,0)]
l.sort(key=lambda x: abs(x))  # inplace, useful to sort on a specific field
doesnt_contains_three = 3 not in l  # "in" operator call __contains__() internally
l.append(7)

# dict operations
keys, values, pairs = di.keys(), di.values(), di.items()
keys_arr, paris_arr = [k for k in di], [(k, v) for k, v in di.items()]
none_val = di.get("absent")  # return None if absent, [] raise KeyError

# maths
log_exp, sqrt_sqr, sin_pi = log(e**2), sqrt(7**2), sin(pi)

# files, "with" statement (try..finally pattern)
with open("file.txt", "r") as f:  # r: read, w: write, a: append
    # call one of those 3 will move the file cursor to the end
    all_file: str = f.read()
    all_lines: list[str] = f.readlines()
    line_by_line: list[str] = [line for line in f]

# string
f_string_ten_is_ten = f"ten is {2*5}"  # string interpolation
multi_line_str_zen_python = """
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
"""

# ternary operator
yes = "yes" if "ten" in f_string_ten_is_ten else "no"

# exceptions, input, output, cast, isinstance
try:
    input_str: str = input("enter a number")
    input_int: int = int(input_str)
    is_int = isinstance(input_int, int)
except ValueError:
    print("error")

# enumerate over iterator with index (lazy generator)
for idx, elt in enumerate(l):
    print(f'elt at {idx} is {elt}')

# Assignment expression (walrus operator) :=
# useful to both check and use a variable inside a scope
positive_logs = [log_e for e in l if (log_e := log(e)) > 0]

# Generators and generator expressions
gen_expr = (factorial(i) for i in range(1, 10))
def gen_factorials(n: int):
    curr = 1
    for i in range(1,n):
        curr *= i
        yield i, curr


# Class, Interface and inheritance
class MyInterface:
    """
        class docstring
    """

    def do(self, i: int):
        """
        function docstring
        :param i:
        :return:
        """
        raise NotImplementedError


class MyImpl(MyInterface):
    def __init__(self):
        super().__init__()

    def do(self, i: int):
        pass


# functions
# static function, private function
class MyClass:
    @staticmethod
    def my_static_method():
        pass

    def __private_func(self):
        pass

# multiple return values (tuple), closure (inner function)
def fctReturnTuple() -> tuple[int,int]:
    def helper():
        return 0, 1
    return helper()

a, b = fctReturnTuple()

# variable positional arguments (*varargs)
# keyword arguments (**kwargs)
def my_fct_all_args(known_positional_arg: str, *varargs, known_named_arg: str, **kwargs):
    print(known_positional_arg)
    print(varargs)
    print(known_named_arg)
    print(kwargs)


# Both calls are equivalents and return:
#   known_positional_arg
#   ('vararg1', 'vararg2')
#   known_named_arg
#   {'kwargs1': 'kwargs1', 'kwargs2': 'kwargs2'}
my_fct_all_args(
    "known_positional_arg",
    "vararg1", "vararg2",
    known_named_arg="known_named_arg",
    kwargs1="kwargs1", kwargs2="kwargs2"
)
my_fct_all_args(
    "known_positional_arg",
    *["vararg1", "vararg2"],
    **{"kwargs1": "kwargs1", "kwargs2": "kwargs2"}
)

# tests, mocks
from unittest import TestCase, main
from unittest.mock import Mock
class MyTestedClassTestCase(TestCase):

    def setUp(self):
        # prepare tests
        pass

    def tearDown(self):
        # clean
        pass

    def test_my_tested_func(self):
        now_func = Mock(spec=datetime.utcnow)
        now_func.return_value = datetime(2019, 6, 5, 15, 45)
        self.assertNotEqual(now_func(), datetime(2019,1,1))

# to execute current test file
if __name__ == '__main__':
    main()



```


## Numpy

```python
import numpy as np

# --- Initialization
from_python_array = np.array([[1, 2, 3], [4, 5, 6]])  # [[1 2 3], [4 5 6]]
zeros = np.zeros((2, 3), dtype=float)  # [[0. 0. 0.], [0. 0. 0.]], "float" is the default
ones = np.ones((2, 3))  # [[1. 1. 1.], [1. 1. 1.]]
memory_not_initialized = np.empty((2, 3))
# 'like' init functions work also with zeros_like, empty_like...
same_shape_as_model = np.zeros_like(from_python_array)  # [[0. 0. 0.], [0. 0. 0.]]
step_by_step = np.arange(2, 9, 2)  # [2, 4, 6, 8], (min, max, step) like python "range"
linear_space = np.linspace(0, 10, 5)  # (min, max, nb_points) [ 0., 2.5, 5., 7.5, 10.]
identity = np.eye(3)  # identity matrix of shape (3, 3)
lower_triangular = np.tri(3)  # triangular matrix of shape (3, 3)

# --- sort
arr = np.array([[1.0, 3.0], [20.0, 10.0]])
arr.sort()  # inplace
sorted_copy = np.sort(arr, axis=-1)  # sort along specified axis, here last (-1) axis

# --- see array characteristics
nb_dimensions = arr.ndim
total_nb_elts = arr.size
tuple_of_size_by_dim = arr.shape

# --- reshape
# -1 to let numpy infer the size on this dim (only one), here it gives a (4, 1, 1) tensor
reshaped = np.reshape(arr, newshape=(-1, 1, 1))

# --- squeeze and expand (wrap, unwrap)
# add dim on specified index: [[1. 3.], [10. 20.]] -> [ [[1. 3.]], [[10. 20.]] ]
expanded = np.expand_dims(arr, axis=1)
# remove specified dims (all by default) where size is 1
# here we remove dim one added by expand_dims -> [[1. 3.], [10. 20.]]
squeezed = np.squeeze(expanded, axis=1)

# --- indexing and slicing
# NB: those operations don't create a new array in memory but just a view
arr = np.array([[11., 12., 13.], [21., 22., 23.], [31., 32., 33.]])

# basic slicing, like python
by_index = arr[1, 2]  # 23
by_slice = arr[0, 1:3:1]  # [12, 13]

# fancy indexing, specific to numpy
by_list_of_indexes = arr[:, [0, 2]]  # [[11. 13.], [21. 23.], [31. 33.]]

# boolean indexing, specific to numpy (use broadcasting)
bool_arr = arr > 21  # [[False False False], [ False  True  True], [ True  True  True]]
flat_when_filtered_on_condition_of_same_shape = arr[arr > 21]  # [22. 23. 31. 32. 33.]
filtered_by_dim = arr[arr.sum(-1) > 40, :]  # [[21. 22. 23.], [31. 32. 33.]]

# assigning with indexing, specific to numpy
modified = np.copy(arr)
modified[:, :] = 1.0  # [[1. 1. 1.], [1. 1. 1.], [1. 1. 1.]]

# --- concat
# concatenate array args must have the same number of dims
row_to_concat = np.expand_dims([41, 42, 43], 0)  # [[41 42 43]]
col_to_concat = np.expand_dims([14, 24, 34], -1)  # [[14] [24] [34]]
with_forth_row = np.concatenate((arr, row_to_concat), axis=0)
with_forth_col = np.concatenate((arr, col_to_concat), axis=-1)

# --- operations
# term by term
term_by_term = (arr + arr - arr) * arr / arr
# usual aggregations
min, sum = arr.min(), arr.sum(axis=-1)  # 11, [36. 66. 96.]
# matrix operations
# more complex linear algebra operations are better dealt with SciPy
transposed = arr.T
dot_product = arr @ arr

# --- broadcasting
# dimensions are compared from right (inner) to left (outer) dimension
# 2 dims are compatibles if they are equals or one is of size 1
broadcast_single_number = arr * 2
broadcast_vec = arr * [1, 10, 100]  # [[11. 120. 1300.], [21. 220. 2300.], [31. 320. 3300.]]
# broadcast (2, 3, 3) with (3, 3), term by term op applied to all (3, 3) inner matrices
wrapped_arr = np.expand_dims(arr, axis=0)
concat_wrapped_arr = np.concatenate((wrapped_arr, wrapped_arr), axis=0)
broadcast_matrix = concat_wrapped_arr * [[1, 10, 100], [2, 20, 200], [3, 30, 300]]
# [[[  11.  120. 1300.]
#   [  42.  440. 4600.]
#   [  93.  960. 9900.]]
#
#  [[  11.  120. 1300.]
#   [  42.  440. 4600.]
#   [  93.  960. 9900.]]]

```