"""Minimal YAML loader for simple configuration files."""
from __future__ import annotations

import ast
from typing import Any, Dict, List


class SimpleYAMLParser:
    """Parse a restricted subset of YAML (mappings, scalars, inline lists)."""

    def parse(self, text: str) -> Any:
        self.stack: List[tuple[int, Any]] = [(-1, {})]
        lines = text.splitlines()
        for raw_line in lines:
            if not raw_line.strip() or raw_line.strip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            while len(self.stack) > 1 and indent <= self.stack[-1][0]:
                self.stack.pop()
            container = self.stack[-1][1]
            if line.startswith("- "):
                value = line[2:].strip()
                parsed = self._parse_value(value)
                if not isinstance(container, list):
                    new_list: List[Any] = []
                    if isinstance(container, dict):
                        raise ValueError("List item without preceding key")
                    container = new_list
                container.append(parsed)
                continue
            if ":" not in line:
                raise ValueError(f"Unsupported line: {line}")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not isinstance(container, dict):
                raise ValueError("Mappings must belong to dict containers")
            if value == "":
                new_dict: Dict[str, Any] = {}
                container[key] = new_dict
                self.stack.append((indent, new_dict))
            else:
                container[key] = self._parse_value(value)
        return self.stack[0][1]

    def _parse_value(self, value: str) -> Any:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        try:
            return ast.literal_eval(value)
        except Exception:
            return value


def load_yaml(path: str) -> Any:
    parser = SimpleYAMLParser()
    with open(path, "r", encoding="utf-8") as fh:
        return parser.parse(fh.read())
