import pandas as pd


def _split_ndc(series):
    """
    Split an NDC column into zero-padded 5-4-2 segments.

    Handles the three 10-digit layouts that show up in raw data by padding
    whichever segment is short:
        4-4-2   0904-7162-61 -> 00904-7162-61
        5-3-2  70010-063-01  -> 70010-0063-01
        5-4-1  68180-3381-1  -> 68180-3381-01
    A 2-segment NDC (labeler-product only) comes back with an empty package
    segment.
    """
    parts = series.astype(str).str.strip().str.split('-', expand=True)
    for i, width in enumerate((5, 4, 2)):
        if i in parts.columns:
            parts[i] = parts[i].fillna('').str.zfill(width)
        else:
            parts[i] = ''
    return parts


def format_ndc_11(df, colname, new_colname='ndc_11'):
    """
    Write the full 11-digit NDC as XXXXX-XXXX-XX into new_colname.

    Example: 0904-7162-61 -> 00904-7162-61
    """
    missing = df[colname].isna()
    parts = _split_ndc(df[colname])
    df[new_colname] = parts[0] + '-' + parts[1] + '-' + parts[2]
    df.loc[missing, new_colname] = pd.NA
    return df


def format_ndc_9(df, colname, new_colname='ndc_9'):
    """
    Write the 9-digit labeler-product NDC as XXXXX-XXXX into new_colname,
    dropping the 2-digit package segment.

    Example: 0904-7162-61 -> 00904-7162
    """
    missing = df[colname].isna()
    parts = _split_ndc(df[colname])
    df[new_colname] = parts[0] + '-' + parts[1]
    df.loc[missing, new_colname] = pd.NA
    return df


def format_ndc(df, colname):
    """
    Format NDC column to 5 digits - 4 digits, in place.
    Pads with leading zeros where needed.

    Example: 42385-902 -> 42385-0902
    """
    return format_ndc_9(df, colname, new_colname=colname)


def remove_zeros(df, colname_list):
    """
    Remove leading zeros from ID.

    Example: 00025 -> 25
    """
    for colname in colname_list:
        df[colname] = df[colname].astype(str).str.replace(r'\.0$', '', regex=True).str.lstrip('0').replace('', '0') 

    return df
