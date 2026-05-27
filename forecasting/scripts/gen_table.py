from pathlib import Path

import fitz
import mlflow
import polars as pl
from matplotlib import pyplot as plt
from weasyprint import HTML

mlflow.set_tracking_uri("sqlite:////app/mlflow.db")
df: pl.DataFrame = pl.from_pandas(mlflow.search_runs(search_all_experiments=True))

df = (
    df.filter(
        pl.col("params.csv_path").str.contains("ETTm1"),
        pl.col("params.irregularity_fraction").cast(float) == 0.7,
    )
    .with_columns(
        duration_s=(pl.col("end_time") - pl.col("start_time")).dt.total_seconds()
    )
    .group_by(Model=pl.col("params.model_type").str.to_uppercase())
    .agg(
        MSE=pl.format(
            "{} +- {}",
            pl.col("metrics.test_mse").mean().round(3),
            pl.col("metrics.test_mse").std().round(3),
        ),
        Time=pl.format(
            "{} +- {}",
            pl.col("duration_s").mean().round(),
            pl.col("duration_s").std().round(),
        ),
    )
)

output_path = Path("/app/outputs/ettm1_res.md")
output_path.parent.mkdir(parents=True, exist_ok=True)
with pl.Config(
    tbl_formatting="MARKDOWN",
    tbl_hide_dataframe_shape=True,
    tbl_hide_column_data_types=True,
):
    output_path.write_text(repr(df))
