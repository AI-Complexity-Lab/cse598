#

from logging import getLogger
from multiprocessing import Value
import os
import io
import re
import sys
import math
import itertools
from collections import OrderedDict
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.calculus.util import AccumBounds
from random import random
from collections import Counter

CLEAR_SYMPY_CACHE_FREQ = 10000

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


def simplify(f, seconds):
    """
    Simplify an expression.
    """
    assert seconds > 0

    @timeout(seconds)
    def _simplify(f):
        try:
            f2 = sp.simplify(f)
            if any(s.is_Dummy for s in f2.free_symbols):
                logger.warning(f"Detected Dummy symbol when simplifying {f} to {f2}")
                return f
            else:
                return f2
        except TimeoutError:
            return f
        except Exception as e:
            logger.warning(f"{type(e).__name__} exception when simplifying {f}")
            return f

    return _simplify(f)


def count_occurrences(expr):
    """
    Count atom occurrences in an expression.
    """
    if expr.is_Atom:
        return {expr: 1}
    elif expr.is_Add or expr.is_Mul or expr.is_Pow:
        assert len(expr.args) >= 2
        result = {}
        for arg in expr.args:
            sub_count = count_occurrences(arg)
            for k, v in sub_count.items():
                result[k] = result.get(k, 0) + v
        return result
    else:
        assert len(expr.args) == 1, expr
        return count_occurrences(expr.args[0])


def count_occurrences2(expr):
    """
    Count atom occurrences in an expression.
    """
    result = {}
    for sub_expr in sp.preorder_traversal(expr):
        if sub_expr.is_Atom:
            result[sub_expr] = result.get(sub_expr, 0) + 1
    return result


def remove_root_constant_terms_t(expr, variables, mode):
    """
    Remove root constant terms from a non-constant SymPy expression.
    """
    variables = variables if type(variables) is list else [variables]
    variables = [str(x) for x in variables]
    assert mode in ["add", "mul", "pow"]
    if not any(str(x) in variables for x in expr.free_symbols):
        return expr
    if mode == "add" and expr.is_Add:
        args = [
            arg
            for arg in expr.args
            if any(str(x) in variables for x in arg.free_symbols) or (arg in [-1])
        ]
        if len(args) == 1:
            expr = args[0]
        elif len(args) < len(expr.args):
            expr = expr.func(*args)
    elif mode == "mul" and expr.is_Mul:
        args = [
            arg
            for arg in expr.args
            if any(x in variables for x in arg.free_symbols) or (arg in [-1])
        ]
        if len(args) == 1:
            expr = args[0]
        elif len(args) < len(expr.args):
            expr = expr.func(*args)
    elif mode == "pow" and expr.is_Pow:
        assert len(expr.args) == 2
        if not any(x in variables for x in expr.args[0].free_symbols):
            return expr.args[1]
        elif not any(x in variables for x in expr.args[1].free_symbols):
            return expr.args[0]
        else:
            return expr
    # elif mode == "singleton" and len(expr.args) == 1:
    #     if expr
    #     if not any(str(x) in variables for x in expr.args[0].free_symbols):
    #         return expr.args[1]

    return expr


def remove_root_constant_terms(expr, variables, mode):
    """
    Remove root constant terms from a non-constant SymPy expression.
    """
    variables = variables if type(variables) is list else [variables]
    assert mode in ["add", "mul", "pow"]
    if not any(x in variables for x in expr.free_symbols):
        return expr
    if mode == "add" and expr.is_Add or mode == "mul" and expr.is_Mul:
        args = [
            arg
            for arg in expr.args
            if any(x in variables for x in arg.free_symbols) or (arg in [-1])
        ]
        if len(args) == 1:
            expr = args[0]
        elif len(args) < len(expr.args):
            expr = expr.func(*args)
    elif mode == "pow" and expr.is_Pow:
        assert len(expr.args) == 2
        if not any(x in variables for x in expr.args[0].free_symbols):
            return expr.args[1]
        elif not any(x in variables for x in expr.args[1].free_symbols):
            return expr.args[0]
        else:
            return expr
    return expr


def remove_mul_const(f, variables):
    """
    Remove the multiplicative factor of an expression, and return it.
    """
    if not f.is_Mul:
        return f, 1
    variables = variables if type(variables) is list else [variables]
    var_args = []
    cst_args = []
    for arg in f.args:
        if any(var in arg.free_symbols for var in variables):
            var_args.append(arg)
        else:
            cst_args.append(arg)
    return sp.Mul(*var_args), sp.Mul(*cst_args)


def extract_non_constant_subtree(expr, variables):
    """
    Extract a non-constant sub-tree from an equation.
    """
    last = expr
    while True:
        last = expr
        # expr = remove_root_constant_terms(expr, variables, 'add')
        expr = remove_root_constant_terms(expr, variables, "mul")
        # expr = remove_root_constant_terms(expr, variables, 'pow')
        n_arg = len(expr.args)
        for i in range(n_arg):
            # if len(expr.args) == 1:
            #      #breakpoint()
            expr = expr.subs(
                expr.args[i], extract_non_constant_subtree(expr.args[i], variables)
            )

        if str(expr) == str(last):
            return expr


# def original_check(expr, variables):
#     """
#     Extract a non-constant sub-tree from an equation.
#     """
#     last = expr
#     while True:
#         last = expr
#         expr = remove_root_constant_terms(expr, variables, "add")
#         # expr = remove_root_constant_terms(expr, variables, "mul")
#         # expr = remove_root_constant_terms(expr, variables, "pow")
#         while len(expr.args) == 1:
#             expr = expr.args[0]
#         if expr == last:
#             return expr


def check_additive_constants(expr, variables):
    """
    Extract a non-constant sub-tree from an equation.
    """
    last = expr
    while True:
        last = expr
        # expr = remove_root_constant_terms(expr, variables, "mul")
        expr = remove_root_constant_terms_t(expr, variables, "add")
        # if expr == None:
        #     return True
        if expr != last:
            return True
        # expr = remove_root_constant_terms(expr, variables, 'pow')
        n_arg = len(expr.args)
        for i in range(n_arg):
            # if len(expr.args) == 1:
            #      #breakpoint()
            if check_additive_constants(expr.args[i], variables) == True:
                return True
            else:
                continue

        if str(expr) == str(last):
            return False


def add_multiplicative_constants(expr, multiplicative_placeholder, unary_operators=[]):
    """
    Traverse the tree in post-order fashion and add multiplicative placeholders
    """

    begin = expr

    if not expr.args:
        if type(expr) == sp.core.numbers.NegativeOne:
            return expr
        else:
            return multiplicative_placeholder * expr
    for sub_expr in expr.args:
        expr = expr.subs(sub_expr,add_multiplicative_constants(sub_expr, multiplicative_placeholder, unary_operators=unary_operators))
    
    if str(type(expr)) in unary_operators:
        expr = multiplicative_placeholder * expr
    return expr


def add_additive_constants(expr, placeholders, unary_operators=[]):
    begin = expr
    if not expr.args:
        if type(expr) == sp.core.numbers.NegativeOne or str(expr) == str(placeholders["cm"]):
            return expr
        else:
            return placeholders["ca"] + expr
    for sub_expr in expr.args:
        expr = expr.subs(sub_expr,add_additive_constants(sub_expr, placeholders, unary_operators=unary_operators))
    
    if str(type(expr)) in unary_operators:
        expr = placeholders["ca"] + expr
    
    return expr

def reindex_coefficients(expr, coefficients):
    """
    Re-index coefficients (i.e. if a1 is there and not a0, replace a1 by a0, and recursively).
    """
    coeffs = sorted(
        [x for x in expr.free_symbols if x in coefficients], key=lambda x: x.name
    )
    for idx, coeff in enumerate(coefficients):
        if idx >= len(coeffs):
            break
        if coeff != coeffs[idx]:
            expr = expr.subs(coeffs[idx], coeff)
    return expr


def reduce_coefficients(expr, variables, coefficients):
    """
    Reduce coefficients in an expression.
    `sqrt(x)*y*sqrt(1/a0)` -> `a0*sqrt(x)*y`
    `x**(-cos(a0))*y**cos(a0)` -> `x**(-a0)*y**a0`
    """
    temp = sp.Symbol("temp")
    while True:
        last = expr
        for a in coefficients:
            if a not in expr.free_symbols:
                continue
            for subexp in sp.preorder_traversal(expr):
                if a in subexp.free_symbols and not any(
                    var in subexp.free_symbols for var in variables
                ):
                    p = expr.subs(subexp, temp)
                    if a in p.free_symbols:
                        continue
                    else:
                        expr = p.subs(temp, a)
                        break
        if last == expr:
            break
    return expr


def simplify_const_with_coeff(expr, coeff):
    """
    Simplify expressions with constants and coefficients.
    `sqrt(10) * a0 * x` -> `a0 * x`
    `sin(a0 + x + 9/7)` -> `sin(a0 + x)`
    `a0 + x + 9` -> `a0 + x`
    """
    assert coeff.is_Atom
    for parent in sp.preorder_traversal(expr):
        if any(coeff == arg for arg in parent.args):
            break
    if not (parent.is_Add or parent.is_Mul):
        return expr
    removed = [arg for arg in parent.args if len(arg.free_symbols) == 0]
    if len(removed) > 0:
        removed = parent.func(*removed)
        new_coeff = (coeff - removed) if parent.is_Add else (coeff / removed)
        expr = expr.subs(coeff, new_coeff)
    return expr


def simplify_equa_diff(_eq, required=None):
    """
    Simplify a differential equation by removing non-zero factors.
    """
    eq = sp.factor(_eq)
    if not eq.is_Mul:
        return _eq
    args = []
    for arg in eq.args:
        if arg.is_nonzero:
            continue
        if required is None or arg.has(required):
            args.append(arg)
    assert len(args) >= 1
    return args[0] if len(args) == 1 else eq.func(*args)


def smallest_with_symbols(expr, symbols):
    """
    Return the smallest sub-tree in an expression that contains all given symbols.
    """
    assert all(x in expr.free_symbols for x in symbols)
    if len(expr.args) == 1:
        return smallest_with_symbols(expr.args[0], symbols)
    candidates = [
        arg for arg in expr.args if any(x in arg.free_symbols for x in symbols)
    ]
    return (
        smallest_with_symbols(candidates[0], symbols) if len(candidates) == 1 else expr
    )


def smallest_with(expr, symbol):
    """
    Return the smallest sub-tree in an expression that contains a given symbol.
    """
    assert symbol in expr.free_symbols
    candidates = [arg for arg in expr.args if symbol in arg.free_symbols]
    if len(candidates) > 1 or candidates[0] == symbol:
        return expr
    else:
        return smallest_with(candidates[0], symbol)


def clean_degree2_solution(expr, x, a8, a9):
    """
    Clean solutions of second order differential equations.
    """
    last = expr
    while True:
        for a in [a8, a9]:
            if a not in expr.free_symbols:
                return expr
            small = smallest_with(expr, a)
            if small.is_Add or small.is_Mul:
                counts = count_occurrences2(small)
                if counts[a] == 1 and a in small.args:
                    if x in small.free_symbols:
                        expr = expr.subs(
                            small,
                            small.func(
                                *[
                                    arg
                                    for arg in small.args
                                    if arg == a or x in arg.free_symbols
                                ]
                            ),
                        )
                    else:
                        expr = expr.subs(small, a)
        if expr == last:
            break
        last = expr
    return expr


def has_inf_nan(*args):
    """
    Detect whether some expressions contain a NaN / Infinity symbol.
    """
    for f in args:
        if f.has(sp.nan) or f.has(sp.oo) or f.has(-sp.oo) or f.has(sp.zoo):
            return True
    return False


def has_I(*args):
    """
    Detect whether some expressions contain complex numbers.
    """
    for f in args:
        if f.has(sp.I):
            return True
    return False


if __name__ == "__main__":

    f = sp.Function("f", real=True)
    x = sp.Symbol("x", positive=True, real=True)
    y = sp.Symbol("y", positive=True, real=True)
    z = sp.Symbol("z", positive=True, real=True)
    a0 = sp.Symbol("a0", positive=True, real=True)
    a1 = sp.Symbol("a1", positive=True, real=True)
    a2 = sp.Symbol("a2", positive=True, real=True)
    a3 = sp.Symbol("a3", positive=True, real=True)
    a4 = sp.Symbol("a4", positive=True, real=True)
    a5 = sp.Symbol("a5", positive=True, real=True)
    a6 = sp.Symbol("a6", positive=True, real=True)
    a7 = sp.Symbol("a7", positive=True, real=True)
    a8 = sp.Symbol("a8", positive=True, real=True)
    a9 = sp.Symbol("a9", positive=True, real=True)

    local_dict = {
        "f": f,
        "x": x,
        "y": y,
        "z": z,
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "a5": a5,
        "a6": a6,
        "a7": a7,
        "a8": a8,
        "a9": a9,
    }

    failed = 0

    #
    # count occurrences
    #

    def test_count_occurrences(infix, ref_counts):
        expr = parse_expr(infix, local_dict=local_dict)
        counts = count_occurrences(expr)
        if set(counts.keys()) != set(ref_counts.keys()) or not all(
            ref_counts[k] == v for k, v in counts.items()
        ):
            print(f"Expression {infix} - Expected: {ref_counts} - Returned: {counts})")
            return False
        return True

    def test_count_occurrences2(infix, _counts):
        expr = parse_expr(infix, local_dict=local_dict)
        counts = count_occurrences2(expr)
        assert set(counts.keys()) == set(_counts.keys())
        if not all(_counts[k] == v for k, v in counts.items()):
            print(f"Expression {infix} - Expected: {_counts} - Returned: {counts})")
            return False
        return True

    tests = [
        ("2", {2: 1}),
        ("2*x", {2: 1, x: 1}),
        ("(2*x)**(3*y+1)", {1: 1, 2: 1, 3: 1, x: 1, y: 1}),
        ("(2*x)**(3*y+x+1)", {1: 1, 2: 1, 3: 1, x: 2, y: 1}),
        ("(2*x)**(3*y+x+1)+a0*x", {1: 1, 2: 1, 3: 1, x: 3, y: 1, a0: 1}),
    ]

    for test in tests:
        failed += not test_count_occurrences(*test)
        failed += not test_count_occurrences2(*test)

    #
    # remove root constant terms
    #

    def test_remove_root_constant_terms(infix, ref_output, mode, variables):
        expr = parse_expr(infix, local_dict=local_dict)
        ref_output = parse_expr(ref_output, local_dict=local_dict)
        output = remove_root_constant_terms(expr, variables, mode)
        if output != ref_output:
            print(
                f"Error when removing constant on expression {infix} with mode {mode} - Expected: {ref_output} - Returned: {output}"
            )
            return False
        return True

    tests = [
        ("x", "x", "add"),
        ("x + 2", "x", "add"),
        ("a0*x + 2", "a0*x", "add"),
        ("x + exp(2)", "x", "add"),
        ("x + exp(2) * x", "x + exp(2) * x", "add"),
        ("x + 2 + a0", "x", "add"),
        ("x + 2 + a0 + z", "x + z", "add"),
        ("x + z", "x + z", "add"),
        ("x + 2", "x + 2", "mul"),
        ("x + z", "x + z", "mul"),
        ("x + z", "x + z", "mul"),
        ("a0 * x", "x", "mul"),
        ("(1 / sqrt(a0)) * x", "x", "mul"),
        ("(3 / sqrt(a0)) * x", "x", "mul"),
        ("(3*a0/a1) * sqrt(x)", "sqrt(x)", "mul"),
        ("exp(x) / sqrt(a0 + 1)", "exp(x)", "mul"),
        ("x + z", "x + z", "mul"),
        ("x + z", "x + z", "mul"),
        ("x + 2", "x + 2", "pow"),
        ("(x + 2) ** 2", "x + 2", "pow"),
        ("(x + 2) ** a0", "x + 2", "pow"),
        ("(x + 2) ** (a0 + 2)", "x + 2", "pow"),
        ("(x + 2) ** (y + 2)", "(x + 2) ** (y + 2)", "pow"),
        ("2 ** (x + 2)", "x + 2", "pow"),
        ("a0 ** (x + 2)", "x + 2", "pow"),
        ("(a0 + 2) ** (x + 2)", "x + 2", "pow"),
        ("(y + 2) ** (x + 2)", "(y + 2) ** (x + 2)", "pow"),
    ]

    for test in tests:
        failed += not test_remove_root_constant_terms(*test, variables=[x, y, z])

    #
    # extract non-constant sub-tree
    #

    def test_extract_non_constant_subtree(infix, ref_output):
        expr = parse_expr(infix, local_dict=local_dict)
        ref_output = parse_expr(ref_output, local_dict=local_dict)
        output = extract_non_constant_subtree(expr, [x, y, z])
        if output != ref_output:
            print(
                f"Error when extracting non-constant sub-tree expression {infix} - Expected: {ref_output} - Returned: {output}"
            )
            return False
        return True

    tests = [
        ("x + sqrt(a0 * x)", "x + sqrt(a0 * x)"),
        ("x + sqrt(a0 * x) + 3", "x + sqrt(a0 * x)"),
        ("x + sqrt(a0 * x) + a1", "x + sqrt(a0 * x)"),
        ("x + sqrt(a0 * x) + a0", "x + sqrt(a0 * x) + a0"),
        ("x + sqrt(a0 * x) + 2 * a0", "x + sqrt(a0 * x) + 2 * a0"),
        ("a0 * x + x + a0", "a0 * x + x + a0"),
        ("(x + sqrt(a0 * x)) ** 2", "x + sqrt(a0 * x)"),
        ("exp(x + sqrt(a0 * x))", "x + sqrt(a0 * x)"),
        ("exp(x + sqrt(a0 * x))", "x + sqrt(a0 * x)"),
    ]

    for test in tests:
        failed += not test_extract_non_constant_subtree(*test)

    #
    # re-index coefficients
    #


    #
    # reduce coefficients
    #

    def test_reduce_coefficients(infix, ref_output):
        expr = parse_expr(infix, local_dict=local_dict)
        ref_output = parse_expr(ref_output, local_dict=local_dict)
        output = reduce_coefficients(
            expr, [x, y, z], [local_dict[f"a{i}"] for i in range(10)]
        )
        if output != ref_output:
            print(
                f"Error when reducing coefficients on expression {infix} - Expected: {ref_output} - Returned: {output}"
            )
            return False
        return True

    tests = [
        ("a0 + 1", "a0"),
        ("a0 + x", "a0 + x"),
        ("1 / sqrt(a0)", "a0"),
        ("1 / (cos(x + sqrt(a0)))", "1 / (cos(x + a0))"),
        ("a0 / (cos(x + sqrt(a0)))", "a0 / (cos(x + sqrt(a0)))"),
        ("sqrt(a0) / (cos(x + sqrt(a0)))", "a0 / (cos(x + a0))"),
        ("ln(a0) / (cos(x + sqrt(a0)))", "ln(a0 ** 2) / (cos(x + a0))"),
        ("ln(a1) / (cos(x + sqrt(a0)))", "a1 / (cos(x + a0))"),
        ("sin(a1) * cos(a0 ** 2 + x)", "a1 * cos(a0 + x)"),
        ("sin(a0) * cos(a0 ** 2 + x)", "sin(sqrt(a0)) * cos(a0 + x)"),
        ("sin(a0 + x) * cos(a0 ** 2 + x)", "sin(sqrt(a0) + x) * cos(a0 + x)"),
        ("sin(a0 + x) * cos(a0 ** 2 + a1)", "sin(a0 + x) * a1"),
        ("sin(a1 + x) * cos(a1 ** 2 + a0)", "sin(a1 + x) * a0"),
        ("sin(sqrt(a0) + x) * a1", "sin(a0 + x) * a1"),
    ]

    for test in tests:
        failed += not test_reduce_coefficients(*test)

    #
    # simplify constants with coefficients
    #

    def test_simplify_const_with_coeff(infix, ref_output, coeff):
        expr = parse_expr(infix, local_dict=local_dict)
        ref_output = parse_expr(ref_output, local_dict=local_dict)
        output = simplify_const_with_coeff(expr, coeff)
        if output != ref_output:
            print(
                f"Error when simplifying constants with coefficient {coeff} on expression {infix} - Expected: {ref_output} - Returned: {output}"
            )
            return False
        return True

    tests = [
        ("sqrt(5) * y * x ** (3 / 2) + 5", "sqrt(5) * y * x ** (3 / 2) + 5", a0),
        ("sqrt(10) * a0 * x", "a0 * x", a0),
        ("sqrt(10) * a0 * x", "sqrt(10) * a0 * x", a1),
        ("2 * a0 * x + 1", "a0 * x + 1", a0),
        ("a0 + tan(x + 5) + 5", "a0 + tan(x + 5)", a0),
        ("a0 + a1 + 5 + tan(x + 5)", "a0 + a1 + tan(x + 5)", a0),
        ("a0 + a1 + 5 + tan(x + 5)", "a0 + a1 + tan(x + 5)", a1),
        ("a0 + x + 9", "a0 + x", a0),
        (
            "9 * a0 * x ** 3 + 36 * a0 * x ** 2/5 + x * cos(x)",
            "a0 * x ** 3 + 4 * a0 * x ** 2/5 + x * cos(x)",
            a0,
        ),
        ("sqrt(10) * cos((a0 + 1) ** 2) * x", "sqrt(10) * cos(a0 ** 2) * x", a0),
        ("2 * a0 * x + 1 - 3 * a0 * cos(x)", "(-2 / 3) * a0 * x + 1 + a0 * cos(x)", a0),
        ("ln(sin(a0 + x + 9 / 7) + 1)", "ln(sin(a0 + x) + 1)", a0),
        ("(a0 + 1) * x ** 2 + x ** 2 + x", "a0 * x ** 2 + x ** 2 + x", a0),
        ("-3 * a0 - 2 * a0 / x + 3 * x + 2", "a0 + 2 * a0 / (3 * x) + 3 * x + 2", a0),
    ]

    for test in tests:
        failed += not test_simplify_const_with_coeff(*test)

    # test results
    if failed == 0:
        print("All tests ran successfully.")
    else:
        print(f"{failed} tests failed!")



class NotCorrectIndependentVariables(Exception):
    pass

class UnknownSymPyOperator(Exception):
    pass

class ValueErrorExpression(Exception):
    pass

class ImAccomulationBounds(Exception):
    pass

class InvalidPrefixExpression(Exception):
    pass

class Generator(object):
    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Pow: "pow",
        sp.exp: "exp",
        sp.log: "ln",
        sp.Abs: 'abs',

        # Trigonometric Functions
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",

        # Trigonometric Inverses
        sp.asin: "asin",
        sp.acos: "acos",
        sp.atan: "atan",

        # Hyperbolic Functions
        sp.sinh: "sinh",
        sp.cosh: "cosh",
        sp.tanh: "tanh",

    }

    OPERATORS = {
        # Elementary functions
        "add": 2,
        "sub": 2,
        "mul": 2,
        "div": 2,
        "pow": 2,
        "inv": 1,
        "pow2": 1,
        "pow3": 1,
        "pow4": 1,
        "pow5": 1,
        "sqrt": 1,
        "exp": 1,
        "ln": 1,
        "abs": 1,

        # Trigonometric Functions
        "sin": 1,
        "cos": 1,
        "tan": 1,

        # Trigonometric Inverses
        "asin": 1,
        "acos": 1,
        "atan": 1,

        # Hyperbolic Functions
        "sinh": 1,
        "cosh": 1,
        "tanh": 1,
        "coth": 1,
    }
    operators = sorted(list(OPERATORS.keys()))
    constants = ["pi", "E"]
    def __init__(self, params):
        self.max_ops = params.max_ops
        self.max_len = params.max_len
        #self.positive = params.positive


        # parse operators with their weights
        
        ops = params.operators.split(",")
        ops = sorted([x.split(":") for x in ops])
        assert len(ops) >= 1 and all(o in self.OPERATORS for o, _ in ops)
        self.all_ops = [o for o, _ in ops]
        self.una_ops = [o for o, _ in ops if self.OPERATORS[o] == 1]
        self.bin_ops = [o for o, _ in ops if self.OPERATORS[o] == 2]
        self.all_ops_probs = np.array([float(w) for _, w in ops]).astype(np.float64)
        self.una_ops_probs = np.array(
            [float(w) for o, w in ops if self.OPERATORS[o] == 1]
        ).astype(np.float64)
        self.bin_ops_probs = np.array(
            [float(w) for o, w in ops if self.OPERATORS[o] == 2]
        ).astype(np.float64)
        self.all_ops_probs = self.all_ops_probs / self.all_ops_probs.sum()
        self.una_ops_probs = self.una_ops_probs / self.una_ops_probs.sum()
        self.bin_ops_probs = self.bin_ops_probs / self.bin_ops_probs.sum()

        assert len(self.all_ops) == len(set(self.all_ops)) >= 1
        assert set(self.all_ops).issubset(set(self.operators))
        assert len(self.all_ops) == len(self.una_ops) + len(self.bin_ops)

        # symbols / elements
        self.variables = OrderedDict({})
        for var in params.variables: 
            self.variables[str(var)] =sp.Symbol(str(var), real=True, nonzero=True)
        self.var_symbols = list(self.variables)
        self.pos_dict = {x:idx for idx, x in enumerate(self.var_symbols)}        
        self.placeholders = {}
        self.placeholders["cm"] = sp.Symbol("cm", real=True, nonzero=True)
        self.placeholders["ca"] = sp.Symbol("ca",real=True, nonzero=True)
        assert 1 <= len(self.variables)
        # We do not no a priori how many coefficients an expression has, so to be on the same side we equal to two times the maximum number of expressions
        self.coefficients = [f"{x}_{i}" for x in self.placeholders.keys() for i in range(2*params.max_len)] 
        assert all(v in self.OPERATORS for v in self.SYMPY_OPERATORS.values())

        # SymPy elements
        self.local_dict = {}
        for k, v in list(
            self.variables.items()
        ):  
            assert k not in self.local_dict
            self.local_dict[k] = v

        digits = [str(i) for i in range(-3, abs(6))]
        self.words = (
            list(self.variables.keys())
            + [
                x
                for x in self.operators
                if x not in ("pow2", "pow3", "pow4", "pow5", "sub", "inv")
            ]
            + digits
        )  


        self.id2word = {i: s for i, s in enumerate(self.words, 4)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        # ADD Start and Finish
        self.word2id["P"] = 0
        self.word2id["S"] = 1
        self.word2id["F"] = 2
        self.id2word[1] = "S"
        self.id2word[2] = "F"

        # ADD Constant Placeholder
        self.word2id["c"] = 3
        self.id2word[3] = "c"

        assert len(set(self.word2id.values())) == len(self.word2id.values())
        assert len(set(self.id2word.values())) == len(self.id2word.values())

        #assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)

        # generation parameters
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)

        # initialize distribution for binary and unary-binary trees
        self.bin_dist = self.generate_bin_dist(params.max_ops)
        self.ubi_dist = self.generate_ubi_dist(params.max_ops)

        # rewrite expressions
        self.rewrite_functions = self.return_rewrite_functions(params)


    @classmethod
    def return_local_dict(cls, variables=None):
        local_dict = {}
        for k, v in list(
            variables.items()
        ):  
            assert k not in local_dict
            local_dict[k] = v
        return local_dict


    @classmethod
    def return_rewrite_functions(cls,params):
        r =  [
            x for x in params.rewrite_functions.split(",") if x != ""
        ]
        assert len(r) == len(set(r))
        assert all(
            x in ["expand", "factor", "expand_log", "logcombine", "powsimp", "simplify"]
            for x in r
        )
        return r

    def generate_bin_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(1, n) = C_n (n-th Catalan number)
            D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
        """
        # initialize Catalan numbers
        catalans = [1]
        for i in range(1, 2 * max_ops + 1):
            catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))

        # enumerate possible trees
        D = []
        for e in range(max_ops + 1):  # number of empty nodes
            s = []
            for n in range(2 * max_ops - e + 1):  # number of operators
                if e == 0:
                    s.append(0)
                elif e == 1:
                    s.append(catalans[n])
                else:
                    s.append(D[e - 1][n + 1] - D[e - 2][n + 1])
            D.append(s)
        return D

    def generate_ubi_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(e, 0) = L ** e
            D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
        """
        # enumerate possible trees
        # first generate the tranposed version of D, then transpose it
        D = []
        D.append([0] + ([self.nl ** i for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(
                    self.nl * s[e - 1]
                    + self.p1 * D[n - 1][e]
                    + self.p2 * D[n - 1][e + 1]
                )
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [
            [D[j][i] for j in range(len(D)) if i < len(D[j])]
            for i in range(max(len(x) for x in D))
        ]
        return D

    def sample_next_pos_ubi(self, nb_empty, nb_ops, rng):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1]
            )
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1]
            )
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity

    def get_leaf(self, curr_leaves, rng):
        if curr_leaves:
            max_idxs = max([self.pos_dict[x] for x in curr_leaves]) + 1
        else:
            max_idxs = 0
        return [list(self.variables.keys())[rng.randint(low=0,high=min(max_idxs+1, len(self.variables.keys())))]]

    def _generate_expr(
        self,
        nb_total_ops,
        rng,
        max_int = 1,
        require_x=False,
        require_y=False,
        require_z=False,
    ):
        """
        Create a tree with exactly `nb_total_ops` operators.
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # create tree
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops, rng)
            if arity == 1:
                op = rng.choice(self.una_ops, p=self.una_ops_probs)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)

            nb_empty += (
                self.OPERATORS[op] - 1 - skipped
            )  # created empty nodes - skipped future leaves
            t_leaves += self.OPERATORS[op] - 1  # update number of total leaves
            l_leaves += skipped  # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = (
                stack[:pos]
                + [op]
                + [None for _ in range(self.OPERATORS[op])]
                + stack[pos + 1 :]
            )

        # sanity check
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is None]) == t_leaves

        leaves = []
        curr_leaves = set()
        for _ in range(t_leaves):
            new_element = self.get_leaf(curr_leaves, rng)
            leaves.append(new_element)
            curr_leaves.add(*new_element)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1 :]
        assert len(leaves) == 0
        return stack
    
    @classmethod
    def write_infix(cls, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == "add":
            return f"({args[0]})+({args[1]})"
        elif token == "sub":
            return f"({args[0]})-({args[1]})"
        elif token == "mul":
            return f"({args[0]})*({args[1]})"
        elif token == "div":
            return f"({args[0]})/({args[1]})"
        elif token == "pow":
            return f"({args[0]})**({args[1]})"
        elif token == "rac":
            return f"({args[0]})**(1/({args[1]}))"
        elif token == "abs":
            return f"Abs({args[0]})"
        elif token == "inv":
            return f"1/({args[0]})"
        elif token == "pow2":
            return f"({args[0]})**2"
        elif token == "pow3":
            return f"({args[0]})**3"
        elif token == "pow4":
            return f"({args[0]})**4"
        elif token == "pow5":
            return f"({args[0]})**5"
        elif token in [
            "sign",
            "sqrt",
            "exp",
            "ln",
            "sin",
            "cos",
            "tan",
            "cot",
            "sec",
            "csc",
            "asin",
            "acos",
            "atan",
            "acot",
            "asec",
            "acsc",
            "sinh",
            "cosh",
            "tanh",
            "coth",
            "sech",
            "csch",
            "asinh",
            "acosh",
            "atanh",
            "acoth",
            "asech",
            "acsch",
        ]:
            return f"{token}({args[0]})"
        elif token == "derivative":
            return f"Derivative({args[0]},{args[1]})"
        elif token == "f":
            return f"f({args[0]})"
        elif token == "g":
            return f"g({args[0]},{args[1]})"
        elif token == "h":
            return f"h({args[0]},{args[1]},{args[2]})"
        elif token.startswith("INT"):
            return f"{token[-1]}{args[0]}"
        else:
            return token
        raise InvalidPrefixExpression(
            f"Unknown token in prefix expression: {token}, with arguments {args}"
        )

    @classmethod
    def add_identifier_constants(cls, expr_list):
        curr = Counter()
        curr["cm"] = 0
        curr["ca"] = 0
        for i in range(len(expr_list)):
            if expr_list[i] == "cm":
                expr_list[i] = "cm_{}".format(curr["cm"])
                curr["cm"] += 1
            if expr_list[i] == "ca":
                expr_list[i] = "ca_{}".format(curr["ca"])
                curr["ca"] += 1
        return expr_list

    def return_constants(self,expr_list):
        #string = "".join(expr_list)
        curr = Counter()
        curr["cm"] = [x for x in expr_list if x[:3] == "cm_"]
        curr["ca"] = [x for x in expr_list if x[:3] == "ca_"]
        return curr
            


    # def sign(self, x):
    #     return ("", "-")[x < 0]
    @classmethod
    def _prefix_to_infix(cls, expr, coefficients=None, variables=None):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in cls.operators:
            args = []
            l1 = expr[1:]
            for _ in range(cls.OPERATORS[t]):  # Arity
                i1, l1 = cls._prefix_to_infix(l1,  coefficients=coefficients, variables=variables)
                args.append(i1)
            return cls.write_infix(t, args), l1
        elif t in coefficients:
            return "{" + t + "}", expr[1:]
        elif (
            t in variables
            or t in cls.constants
            or t == "I"
        ):
            return t, expr[1:]
        else: #INT
            val = expr[0]
            return str(val), expr[1:]

    def _prefix_to_edges(self, expr):
        t = expr[0][1]
        edges = []
        li = expr[1:]
        if t in self.operators:
            args = []
            for _ in range(self.OPERATORS[t]):
                new_edge = [expr[0][0], li[0][0]]
                edges.append(new_edge)
                inner_edges, li = self._prefix_to_edges(li)
                edges.extend(inner_edges)
        return edges, li


    @classmethod
    def prefix_to_infix(cls, expr, coefficients=None, variables=None):
        """
        Prefix to infix conversion.
        """
        p, r = cls._prefix_to_infix(expr, coefficients=coefficients, variables=variables)
        if len(r) > 0:
            raise InvalidPrefixExpression(
                f'Incorrect prefix expression "{expr}". "{r}" was not parsed.'
            )
        return f"({p})"

    @classmethod
    def rewrite_sympy_expr(cfg, expr, rewrite_functions=None):
        """
        Rewrite a SymPy expression.
        """
        expr_rw = expr
        for f in rewrite_functions:
            if f == "expand":
                expr_rw = sp.expand(expr_rw)
            elif f == "factor":
                expr_rw = sp.factor(expr_rw)
            elif f == "expand_log":
                expr_rw = sp.expand_log(expr_rw, force=True)
            elif f == "logcombine":
                expr_rw = sp.logcombine(expr_rw, force=True)
            elif f == "powsimp":
                expr_rw = sp.powsimp(expr_rw, force=True)
            elif f == "simplify":
                expr_rw = simplify(expr_rw, seconds=1)
        return expr_rw

    @classmethod
    def infix_to_sympy(cls, infix, variables, rewrite_functions, no_rewrite=False):
        """
        Convert an infix expression to SymPy.
        """
        try:
            expr = parse_expr(infix, evaluate=True, local_dict=cls.return_local_dict(variables))
        except ValueError:
            raise ImAccomulationBounds
        if expr.has(sp.I) or expr.has(AccumBounds):
            raise ValueErrorExpression
        if not no_rewrite:
            expr = cls.rewrite_sympy_expr(expr, rewrite_functions)
        return expr

    @classmethod
    def _sympy_to_prefix(cls, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)
    
        assert (
            (op == "add" or op == "mul")
            and (n_args >= 2)
            or (op != "add" and op != "mul")
            and (1 <= n_args <= 2)
        )

        # square root
        if (
            op == "pow"
            and isinstance(expr.args[1], sp.Rational)
            and expr.args[1].p == 1
            and expr.args[1].q == 2
        ):
            return ["sqrt"] + Generator.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += Generator.sympy_to_prefix(expr.args[i])

        return parse_list

    @classmethod
    def sympy_to_prefix(cls,expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]  # self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return (
                ["div"] + [str(expr.p)] + [str(expr.q)]
            )  # self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ["E"]
        elif expr == sp.pi:
            return ["pi"]
        elif expr == sp.I:
            return ["I"]
        # SymPy operator
        for op_type, op_name in cls.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return cls._sympy_to_prefix(op_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def process_equation(self, infix):
        f = self.infix_to_sympy(infix, self.variables, self.rewrite_functions)

        
        symbols = set([str(x) for x in f.free_symbols])
        if not symbols:
            raise NotCorrectIndependentVariables()
            #return None, f"No variables in the expression, skip"
        for s in symbols:
            if not len(set(self.var_symbols[:self.pos_dict[s]]) & symbols) == len(self.var_symbols[:self.pos_dict[s]]):
                raise NotCorrectIndependentVariables()
                #return None, f"Variable {s} in the expressions, but not the one before"
        
        f = remove_root_constant_terms(f, list(self.variables.values()), 'add')
        f = remove_root_constant_terms(f, list(self.variables.values()), 'mul')
        f = add_multiplicative_constants(f, self.placeholders["cm"], unary_operators=self.una_ops)
        f = add_additive_constants(f, self.placeholders, unary_operators=self.una_ops)

        return f

    def generate_equation(self, rng):
        """
        Generate pairs of (function, primitive).
        Start by generating a random function f, and use SymPy to compute F.
        """
        nb_ops = rng.randint(3, self.max_ops + 1)
        f_expr = self._generate_expr(nb_ops, rng, max_int=1)
        infix = self.prefix_to_infix(f_expr, coefficients=self.coefficients, variables=self.variables)
        f = self.process_equation(infix)
        f_prefix = self.sympy_to_prefix(f)
        # skip too long sequences
        if len(f_expr) + 2 > self.max_len:
            raise ValueErrorExpression("Sequence longer than max length")
            #return None, "Sequence longer than max length"

        # skip when the number of operators is too far from expected
        real_nb_ops = sum(1 if op in self.OPERATORS else 0 for op in f_expr)
        if real_nb_ops < nb_ops / 2:
            raise ValueErrorExpression("Too many operators")
            #return None, "Too many operators"

        if f == "0" or type(f) == str:
            raise ValueErrorExpression("Not a function")
            #return None, "Not a function"
        
        sy = f.free_symbols
        variables = set(map(str, sy)) - set(self.placeholders.keys())
        return f_prefix, variables



   





