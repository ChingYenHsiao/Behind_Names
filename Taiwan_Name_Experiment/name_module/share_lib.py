import numpy as np
import pandas as pd


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    na_list = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ",props[col].dtype)

            # make variables for Int, max and min
            is_int = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                na_list.append(col)
                props[col].fillna(mn-1, inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if -0.01 < result < 0.01:
                is_int = True

            # Make Integer/unsigned Integer datatypes
            if is_int:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            # print("dtype after: ",props[col].dtype)
            # print("******************************")

    # Print final result
    # print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100*mem_usg/start_mem_usg, "% of the initial size")
    return props, na_list


def is_number(string):
    """ Check string is number or not.
    Args:
        string : target string

    Returns:
        bool: is number or not
    """
    if isinstance(string, (int, float)):
        return True
    try:
        return string.isdecimal()
    except ValueError:
        return False


def is_chinese(string):
    """  Check string only contain chinese or not.
    Args:
        string : target string

    Returns:
        bool: is chinese or not
    """
    try:
        for uchar in string:
            if not('\u4e00' <= uchar <= '\u9fff'):
                return False
        return True
    except ValueError:
        return False


def restore_df_dtypes(df, int8_col=None, int16_col=None, int32_col=None, int64_col=None,
                      float32_col=None, float64_col=None, string_col=None):
    """ Set columns object type to original type.
    Args:
        df (pd.DataFrame): target dataframe
        int8_col (list, optional): target int8 column. Defaults to None.
        int16_col (list, optional): target int16 column. Defaults to None.
        int32_col (list, optional): target int32 column. Defaults to None.
        int64_col (list, optional): target int32 column. Defaults to None.
        float32_col (list, optional): target float32 column. Defaults to None.
        float64_col (list, optional): target float64 column. Defaults to None.
        string_col (list, optional): target string column. Defaults to None.
    """
    type_dict = {}
    run_df_dtypes = dict(int8=int8_col, int16=int16_col, int32=int32_col,
                         int64=int64_col, float32=float32_col, float64=float64_col, string=string_col)
    for convert_type, cols in run_df_dtypes.items():
        if cols is not None:
            for col in cols:
                if col in df.columns:
                    type_dict[col] = convert_type
    df = df.astype(type_dict)
    return df
