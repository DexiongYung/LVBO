import pandas as pd
from plot.args_common import get_common_args


def get_XYZ():
    parser = get_common_args()
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)

    X = df["0"].to_numpy()
    Y = df["1"].to_numpy().T
    Z = df["pageSuffix=pageOne,placementWidgetName=sp_phoneapp_search_mtf,KPI=impressionsLift"].to_numpy()

    return X, Y, Z
