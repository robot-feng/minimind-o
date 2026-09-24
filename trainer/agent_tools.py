"""Sandboxed toy tools and verifiable rewards for MiniMind-O Agent RL."""

import ast
import json
import operator
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from trainer.rl_utils import heuristic_response_reward, repetition_penalty


def safe_math_eval(expression):
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                  ast.Mod: operator.mod, ast.Pow: operator.pow}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 16 or abs(left) > 1e6):
                raise ValueError("exponent is out of bounds")
            return operations[type(node.op)](left, right)
        raise ValueError("only numeric arithmetic expressions are allowed")

    text = str(expression).strip()
    if not text or len(text) > 256:
        raise ValueError("expression is empty or too long")
    return evaluate(ast.parse(text, mode="eval").body)


def parse_tool_calls(text):
    calls = []
    for raw in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(raw.strip())
        except (TypeError, json.JSONDecodeError):
            continue
        if "function" in call and isinstance(call["function"], dict):
            call = call["function"]
        calls.append(call)
    return calls


def tool_names(tools):
    names = set()
    for tool in tools or []:
        function = tool.get("function", tool)
        if function.get("name"):
            names.add(function["name"])
    return names


def execute_tool(name, arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    try:
        if name == "calculate_math":
            return {"result": str(safe_math_eval(arguments.get("expression", "")))}
        if name == "get_current_time":
            zone = ZoneInfo(arguments.get("timezone", "Asia/Shanghai"))
            return {"datetime": datetime.now(zone).isoformat(timespec="seconds"), "timezone": str(zone)}
        if name == "get_current_weather":
            return {"location": arguments.get("location", ""), "condition": "sunny", "temperature_c": 22}
        if name == "unit_converter":
            factors = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462,
                       "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048}
            key = f"{arguments.get('from_unit', '').lower()}_{arguments.get('to_unit', '').lower()}"
            if key not in factors:
                return None
            return {"result": float(arguments["value"]) * factors[key]}
        if name == "get_exchange_rate":
            rates = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12,
                     ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79}
            pair = (arguments.get("from_currency", "").upper(), arguments.get("to_currency", "").upper())
            return {"from": pair[0], "to": pair[1], "rate": rates[pair]} if pair in rates else None
        if name == "translate_text":
            examples = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好",
                        ("I love programming", "chinese"): "我喜欢编程"}
            key = (arguments.get("text", ""), arguments.get("target_language", "").lower())
            return {"translated_text": examples.get(key, key[0])}
        return None
    except (ArithmeticError, KeyError, TypeError, ValueError):
        return None


def validate_gt_in_text(text, ground_truth):
    raw = str(text)
    normalized = raw.replace(",", "")
    numbers = [float(value) for value in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])", normalized)]
    found = set()
    for expected in ground_truth or []:
        value = str(expected).strip()
        if not value:
            continue
        if value.lower() in raw.lower():
            found.add(expected)
        elif re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value.replace(",", "")):
            target = float(value.replace(",", ""))
            if any(abs(target - number) < 1e-6 for number in numbers):
                found.add(expected)
    return found


def calculate_agent_reward(final_text, turn_texts, tools, ground_truth, unfinished=False, reward_model=None, prompt=""):
    calls = [call for text in turn_texts for call in parse_tool_calls(text)]
    if not calls:
        reward = heuristic_response_reward(final_text)
        if reward_model:
            reward += reward_model.score(prompt, final_text)
        return max(min(reward, 3.0), -3.0)

    valid_names = tool_names(tools)
    valid_calls = 0
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        required = {"calculate_math": ("expression",), "unit_converter": ("value", "from_unit", "to_unit"),
                    "get_current_weather": ("location",), "get_current_time": (),
                    "get_exchange_rate": ("from_currency", "to_currency"),
                    "translate_text": ("text", "target_language")}.get(name, None)
        valid_calls += int(name in valid_names and required is not None and all(args.get(key) is not None for key in required))
    gap = abs(valid_calls - len(ground_truth or [])) + max(0, len(calls) - valid_calls)
    reward = 0.5 if gap == 0 else -0.5 * gap
    final_answer = "" if unfinished else (final_text.split("</tool_call>")[-1].strip() or final_text)
    if ground_truth:
        reward += 2.5 * len(validate_gt_in_text(final_answer, ground_truth)) / len(ground_truth)
    if unfinished:
        reward -= 0.5
    reward -= repetition_penalty(final_answer or final_text)
    return max(min(reward, 3.0), -3.0)
