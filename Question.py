from typing import Union


class Question():
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, to_match: Union[str, int]):
        if isinstance(self.value, str):
            return self.value == to_match
        elif isinstance(self.value, (int, float)):
            return self.value > to_match

    def __repr__(self) -> str:
        condition = None
        if isinstance(self.value, str):
            condition = "=="
        elif isinstance(self.value, (int, float)):
            condition = ">"
        return f"Is {self.column} {condition} {self.value}?"
