import random 
import pandas as pd 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
Question_PATH = BASE_DIR / "data2" / "leetcode_dataset - lc.csv"

data = pd.read_csv(Question_PATH)
unique_topics = [
    'Array',
    'Backtracking',
    'Bit Manipulation',
    'Binary Search',
    'Binary Indexed Tree',
    'Binary Search Tree',
    'Brainteaser',
    'Breadth-first Search',
    'Depth-first Search',
    'Dequeue',
    'Design',
    'Divide and Conquer',
    'Dynamic Programming',
    'Geometry',
    'Graph',
    'Greedy',
    'Hash Table',
    'Heap',
    'Line Sweep',
    'Linked List',
    'Math',
    'Meet in the Middle',
    'Memoization',
    'Minimax',
    'OOP',
    'Ordered Map',
    'Queue',
    'Random',
    'Recursion',
    'Rejection Sampling',
    'Reservoir Sampling',
    'Rolling Hash',
    'Segment Tree',
    'Sliding Window',
    'Sort',
    'Stack',
    'String',
    'Suffix Array',
    'Topological Sort',
    'Tree',
    'Trie',
    'Two Pointers',
    'Union Find'
]


def get_matchs(x):
    tops = random.sample(unique_topics,x)
    ques = []
    for tp in tops:
        d = data[data["related_topics"] == tp]
        if d.shape[0] >0:
            ques.append(int(d.iloc[0]["id"])-1)
        else:
            ques.append(50)
    return ques 

