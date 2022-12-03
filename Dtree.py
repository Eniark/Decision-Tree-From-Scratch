import numpy as np
from Node import Node
from Question import Question
from typing import Union


# True -> left node
# False -> right node
class DecisionTree:
    def __init__(self, dataset: list[list], col_names: list[str], y_column_number: int, print_tree=False) -> None:
        self.dataset = dataset
        self.col_names = col_names
        X, y = self.Xy_split(y_column_number)
        self.root = Node(X, y)
        gini = self.get_gini_impurity(self.root)
        self.root.gini_impurity = gini
        self.build_tree(self.root)
        if print_tree:
            print("Root Node:")
            self.print_tree(self.root)

    def Xy_split(self, y_column_number: int) -> Union[list[list], list]:
        Xs = []
        ys = []
        for lst in self.dataset:
            y = lst.pop(y_column_number - 1)
            Xs.append(lst)
            ys.append(y)
        return Xs, ys

    def build_tree(self, node: Node) -> Node:
        best_partition_criterion = self.find_best_partition(node)
        if best_partition_criterion['information_gain'] == 0:
            return node
        question = best_partition_criterion['question']
        X_l, y_l, X_r, y_r = node.split(
            question, self.col_names.index(question.column))

        node.question = question

        node_left = Node(X_l, y_l)
        node_right = Node(X_r, y_r)

        node_left__gini_imp = self.get_gini_impurity(node_left)
        node_right__gini_imp = self.get_gini_impurity(node_right)

        node_left.gini_impurity = node_left__gini_imp
        node.left = node_left

        node_right.gini_impurity = node_right__gini_imp
        node.right = node_right

        if node_left__gini_imp != 0:
            return self.build_tree(node_left)
        if node_right__gini_imp != 0:
            return self.build_tree(node_right)
        return node

    @staticmethod
    def get_column_from_list(data: list[list], col_id: int) -> list[str]:
        return [i[col_id] for i in data]

    def find_best_partition(self, node: Node) -> dict:
        ginis = []

        for idx in range(len(node.X[0])):
            col = DecisionTree.get_column_from_list(node.X, idx)
            col_type = 'numeric' if isinstance(col[0], (float, int)) else 'str'
            if col_type == 'numeric':
                min_X, max_X = min(col), max(col)
                split_values = np.linspace(min_X, max_X, 20)
            else:
                split_values = set(col)

            for feature in split_values:
                gini_object = {}
                question = Question(self.col_names[idx], feature)
                X_l, y_l, X_r, y_r = node.split(question, idx)
                node_left = Node(X_l, y_l)  # weird
                node_right = Node(X_r, y_r)
                gini_left = self.get_gini_impurity(node_left)
                gini_right = self.get_gini_impurity(node_right)
                weighted_gini_impurity = len(X_l)/len(node.X) * gini_left + \
                    len(X_r)/len(node.X) * gini_right
                information_gain = node.gini_impurity - weighted_gini_impurity

                gini_object['question'] = question
                gini_object['information_gain'] = round(information_gain, 3)
                ginis.append(gini_object)

        best_criterion_to_split = sorted(
            ginis, key=lambda x: x['information_gain'], reverse=True)[0]
        return best_criterion_to_split

    def get_gini_impurity(self, node: Node) -> float:
        length_of_subset = len(node.y)
        counts_of_ys = node.get_counts_of_each_target()
        node.values = counts_of_ys
        node.samples = sum(node.values.values())
        gini_idx = 0

        for unique_target in node.get_uniques():
            proba = counts_of_ys[unique_target]/length_of_subset
            gini_idx += proba**2

        gini_impurity = 1 - gini_idx
        return round(gini_impurity, 3)

    def predict(self, X_test, node=None):
        if not node:
            node = self.root
        question = node.question
        if question:
            col_idx = self.col_names.index(question.column)
            if question.match(X_test[col_idx]):
                if node.left:
                    return self.predict(X_test=X_test, node=node.left)
            else:
                if node.right:
                    return self.predict(X_test=X_test, node=node.right)
        else:
            probability_result = {}
            for key in node.values:
                probability_result[key] = node.values[key]/node.samples
            return probability_result

    def print_tree(self, node: Node, level: int = 0) -> None:
        print(f'{level}: {node}')
        if node.left:
            print('Left Node:')
            self.print_tree(node.left, level=level+1)
        if node.right:
            print('Right Node:')
            self.print_tree(node.right, level=level+1)


col_names = ["Color", "Diameter", "Fruit"]
data = [
    ["Red", 1, "Grape"],
    ["Green", 3, "Apple"],
    ["Red", 1.5, "Apple"],
    ["Red", 2, "Grape"],
    ["Yellow", 1, "Lemon"],
    ['Yellow', 1, 'Apple']
]

X_test = ['Yellow', 1]
DTree = DecisionTree(data, col_names, 3, True)
print(DTree.predict(X_test))
