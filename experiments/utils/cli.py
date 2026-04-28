"""File implements utility functions for cli experiments with Omegaconf."""

from __future__ import absolute_import, division, print_function

import ast
import math
import operator
from typing import Any

from omegaconf import OmegaConf


def _math_eval(expr: str) -> Any:
    """Safely evaluate simple math expressions (no globals)."""

    def checkmath(func_name, *args):
        allowed_funcs = [x for x in dir(math) if "__" not in x]
        if func_name in ("int", "float"):
            return {"int": int, "float": float}[func_name](*args)
        if func_name not in allowed_funcs:
            raise SyntaxError(f"Unknown func {func_name}()")
        return getattr(math, func_name)(*args)

    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
    }

    un_ops = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return bin_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return un_ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SyntaxError("Only named functions allowed")
            func_name = node.func.id
            args = [_eval(arg) for arg in node.args]
            return checkmath(func_name, *args)
        raise SyntaxError(f"Unsupported syntax: {type(node)}")

    val = _eval(tree.body)
    return int(val) if isinstance(val, float) and val.is_integer() else val


def math_resolver(expr: str) -> Any:
    """OmegaConf resolver that evaluates safe math expressions."""

    return _math_eval(expr.strip())


def safe_eval_resolver(expr: str) -> Any:
    """Backward-compatible resolver that evaluates safe math expressions."""

    return _math_eval(expr.strip())


def join_str(values, fmt=".4f", sep="_"):
    """Join a list of numbers using a given format string and separator.

    Args:
        values (list): List of numbers to format.
        fmt (str): Format string, either Python-style ("{:.2f}") or printf-style ("%.2f").
        sep (str): Separator between elements.

    Returns:
        str: Formatted and joined string.
    """
    formatted = []
    for v in values:
        try:
            # Determine type based on format
            if "d" in fmt or "i" in fmt:
                v = int(round(float(v)))  # coerce to int safely
            else:
                v = float(v)

            # Python-style formatting
            if fmt.startswith("{") and fmt.endswith("}"):
                formatted.append(fmt.format(v))
            else:
                # printf-style formatting
                formatted.append(fmt % v)

        except Exception as e:
            raise ValueError(f"Invalid format '{fmt}' for value {v}: {e}")
    return sep.join(formatted)


_TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
_FALSE_STRINGS = {"false", "0", "no", "n", "off"}


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in _TRUE_STRINGS:
            return True
        if val in _FALSE_STRINGS:
            return False
    raise ValueError(f"Cannot interpret '{value}' as boolean.")


def bool_str(flag, true_value, false_value=""):
    """Return ``true_value`` when ``flag`` is truthy, otherwise
    ``false_value``."""
    return str(true_value) if _to_bool(flag) else str(false_value)


def select_if(cond: bool, true_val, false_val):
    """Return `true_val` if cond is True, else `false_val`."""
    return true_val if cond else false_val


def setup_resolvers():
    OmegaConf.register_new_resolver("eval", safe_eval_resolver)
    OmegaConf.register_new_resolver("math", math_resolver)
    OmegaConf.register_new_resolver("join_str", join_str)
    OmegaConf.register_new_resolver("fstr", lambda val, fmt: format(val, fmt))
    OmegaConf.register_new_resolver("bool_str", bool_str)
    OmegaConf.register_new_resolver("select_if", select_if)
