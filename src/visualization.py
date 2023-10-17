import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Dict
import seaborn as sns
def difference_between_important_features(feature_importances_A: pd.DataFrame, feature_importances_B: pd.DataFrame, 
                   A_name: str= None, B_name: str= None, top_k: int = 20, figsize: Union[Tuple, List]=(4, 8), font_size: int = 15, display_feature_name: Dict[str, str] = None) -> Tuple[plt.figure, plt.axis]:
    """_summary_

    Args:
        feature_importances_A (pd.DataFrame): _description_
        feature_importances_B (pd.DataFrame): _description_
        top_k (int, optional): _description_. Defaults to 20.
        figsize (Union[Tuple, List], optional): _description_. Defaults to (4, 8).
        display_feature_name (Dict[str, str], optional): _description_. Defaults to None.

    Returns:
        Tuple[plt.figure, plt.axis]: _description_
    """
    fig = plt.figure(figsize=figsize)  #(15, 20), (4,8), (5,8)
    ax1 = fig.add_subplot(111)
    
    if display_feature_name is None:
        display_feature_name = {}
        for feature in feature_importances_A["Features"]:
            display_feature_name[feature] = feature
            
    for feature in feature_importances_A["Features"]:
        idx = feature_importances_A["Features"] == feature
        feature_importances_A["Features"][idx] = display_feature_name[feature]
    
    for feature in feature_importances_B["Features"]:
        idx = feature_importances_B["Features"] == feature
        feature_importances_B["Features"][idx] = display_feature_name[feature]
    
    top_k_A = {v: k for k, v in dict(dict(feature_importances_A["Features"].iloc[:top_k])).items()}
    top_k_B = {v: k for k, v in dict(dict(feature_importances_B["Features"].iloc[:top_k])).items()}
    
    feature_list = set(np.concatenate((list(top_k_A.keys()), list(top_k_B.keys()))))
    
    x_length = [0.01, 0.99]
    for feature in feature_list:
        if feature in top_k_A.keys():
            if feature in top_k_B.keys():
                ax1.plot(x_length, [top_k - top_k_A[feature], top_k - top_k_B[feature]], alpha = 0.5, c="purple")
            else:
                ax1.plot(x_length, [top_k - top_k_A[feature], 0], alpha = 0.5, c="red")
        else:
            ax1.plot(x_length, [0, top_k - top_k_B[feature]], alpha=0.5, c = "dodgerblue")
    
    fontsize=font_size
    plt.title(A_name, loc='left')
    ax1.set_ylim(-0.5, top_k+1)
    ax1.set_xlim(0, 1)
    ax1.set_yticks([i for i in range(top_k, -1, -1)])
    ax1.set_yticklabels(np.concatenate((list(top_k_A.keys()), ["Other features"])), fontsize=fontsize)
    ax1.get_xaxis().set_visible(False)

    ax2 = ax1.twinx()  
    plt.title(B_name, loc='right')
    ax2.set_ylim(-0.5, top_k+1)
    ax2.set_yticks([i for i in range(top_k, -1, -1)])
    ax2.set_yticklabels(np.concatenate((list(top_k_B.keys()), ["Other features"])), fontsize=fontsize)
    plt.show()