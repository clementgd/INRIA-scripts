from typing import List, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import namedtuple


#
#
# Core function
#
#


ValuesGroup = namedtuple("ValuesGroup", ["lower_bound", "higher_bound", "size"])

# Uses binary search for faster execution 
def get_groups(values: np.array, eps: int, min_group_size: int) -> List[ValuesGroup]:
    n = len(values)
    sorted_values = np.sort(values)
    
    groups = []
    curr_idx = 0 # Idx of the beginning of the current group
    curr_val = sorted_values[0]
    end_val = sorted_values[n - 1]
    
    while end_val - curr_val > eps:
        begin = curr_idx + 1        
        end = n - 1
        
        while end > begin + 1:
            middle = begin + (end - begin) // 2
            delta = sorted_values[middle] - curr_val
            if delta > eps:
                end = middle # End is always >. Doesn't hold on the last iteration
            else :
                begin = middle # Begin is always <=
        
        group_size = begin - curr_idx + 1
        # print(f"curr_idx : {curr_idx}, begin : {begin}, group size : {group_size} "
        #       f"curr val : {curr_val}, begin val : {sorted_values[begin]}, end val : {sorted_values[end]} "
        #       f"delta begin : {sorted_values[begin] - curr_val}, delta end : {sorted_values[end] - curr_val}")
        if group_size >= min_group_size:
            groups.append(ValuesGroup(lower_bound = curr_val, higher_bound = sorted_values[begin], size = group_size))
            
        curr_idx = begin + 1
        curr_val = sorted_values[curr_idx]
        
    groups.append(ValuesGroup(lower_bound = curr_val, higher_bound = end_val, size = n - curr_idx))
            
    return groups


def get_groups_no_binary_search(values: np.array, eps: int, min_group_size: int):
    n = len(values)
    sorted_values = np.sort(values)
    
    clusters = []
    curr_idx = 0
    curr_val = sorted_values[0]
    # current_group = [sorted_values[0]]

    for i, val in enumerate(sorted_values):
        if val - curr_val > eps:
            group_size = i - curr_idx
            if group_size >= min_group_size:
                clusters.append((curr_val, sorted_values[i - 1], group_size))
            curr_idx = i
            curr_val = val
            
    clusters.append((curr_val, sorted_values[n - 1], n - curr_idx))
            
    return clusters



#
#
# Helper functions
#
#

def print_groups(groups: List[ValuesGroup]):
    for i, g in enumerate(groups):
        print(f"Group {i} -- bounds : {g[0], g[1]}, width : {'{:.3e}'.format(g[1] - g[0])}, size : {g[2]}")


def extract_group(data, group: ValuesGroup, variable = None):
    if isinstance(data, pd.DataFrame):
        if variable is None:
            raise Exception("Need to specify a variable when extracting group from DataFrame")
        return data.loc[(data[variable] >= group.lower_bound) & (data[variable] <= group.higher_bound)]
    elif isinstance(data, np.ndarray):
        # Assumed to be a numpy array
        return data[(data >= group.lower_bound) & (data <= group.higher_bound)]
    else:
        raise Exception("extract_group does not support data types other than pd.DataFrame and np.array")
    
    

def extract_group_df(df: pd.DataFrame, variable: str, group: ValuesGroup):
    return df.loc[(df[variable] >= group.lower_bound) & (df[variable] <= group.higher_bound)]


def plot_group(values: np.array, group: ValuesGroup):
    filtered_values = np.sort(values[(values >= group.lower_bound) & (values <= group.higher_bound)])
    plt.plot(range(len(filtered_values)), filtered_values)
    plt.show()