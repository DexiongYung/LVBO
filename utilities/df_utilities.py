import torch
import pandas as pd


def pd_to_tensor(data: pd.DataFrame, qual_col_names: list, quant_col_names: list, target: str):
    X_cont = torch.Tensor(data[quant_col_names].to_numpy())
    X_qual = torch.Tensor(data[qual_col_names].to_numpy())
    Y = torch.Tensor(data[target].to_numpy())

    return X_cont, X_qual, Y

