from Question import Question


class Node:
    def __init__(self, X: list[list], y: list[str]):
        self.X = X
        self.y = y
        self.left = None
        self.right = None
        self.gini_impurity = None
        self.question = None
        self.values = None

    def get_uniques(self) -> set:
        return set(self.y)

    def get_counts_of_each_target(self) -> dict:
        counts = {}
        for i in self.y:
            if i not in counts:
                counts[i] = 1
            else:
                counts[i] += 1
        return counts

    def split(self, question: Question, feature_id: int) -> None:
        X_left = []
        y_left = []
        X_right = []
        y_right = []
        for row_idx, x_row in enumerate(self.X):
            if question.match(x_row[feature_id]):
                X_left.append(x_row)
                y_left.append(self.y[row_idx])
            else:
                X_right.append(x_row)
                y_right.append(self.y[row_idx])

        return X_left, y_left, X_right, y_right

    def __str__(self) -> None:
        st = f"""\tQuestion: {self.question}\n\tGini_Impurity = {self.gini_impurity}\n\tSamples = {self.samples}\n\tValue = [{self.values}]\n"""
        return st
