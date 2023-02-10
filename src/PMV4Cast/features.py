"""
This module should contain project specific feature engineering functionality.

You should avoid engineering features in a notebook as it is not transferable later if you want to automate the
process. Add functions here to create your features, such functions should include those to generate specific features
along with any more generic functions.

Consider moving generic functions into the shared statoilds package.
"""
import numpy as np
import pandas as pd
import scipy.stats as st


def my_feature_xxx(df: pd.DataFrame):
    """
    Description goes here.
    You might also add additional arguments such as column etc...
    Would be nice with some test cases also :)

    Args:
        df: Dataframe upon which to operate

    Returns:
        A dataframe with the Xxx feature appended
    """

    # CODE HERE

    return df


def quantize(mean, var, q):
    z_score = st.norm.ppf(q)
    return mean + z_score * var
