from math import pi

import polars as pl


def temporal_cyclic_features(timecol: pl.Expr, period: str):
    start = timecol.dt.truncate(period)
    end = start.dt.offset_by(period)
    normalized = (timecol - start) / (end - start)
    cos = (normalized * 2 * pi).cos().alias(f"{period}_cos")
    sin = (normalized * 2 * pi).sin().alias(f"{period}_sin")
    return cos, sin


def extract_time_feats(time_col: str, df: pl.DataFrame):

    time_feat_exprs: list[pl.Expr] = []
    for period in ["1d", "1w", "1m", "1q", "1y"]:
        time_feat_exprs.extend(temporal_cyclic_features(pl.col(time_col), period))
    times = df.select(time_feat_exprs).to_numpy()
    return times
