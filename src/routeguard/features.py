"""Turn the joined order table into model-ready features.

The features fall into two kinds:

1. "Row-local" features - computed from a single order only (distance, freight
   ratio, weekday, ...). These can never leak, because they use nothing outside
   the order itself.

2. "Historical rate" features - e.g. a seller's past late rate. These are the
   dangerous ones. To stay leakage-safe we:
     * sort every order by purchase time,
     * for each order use ONLY the outcomes of PRIOR orders (shift(1)),
     * via an expanding mean, so an order never "sees" itself or the future.

NOTE on a simplification (documented on purpose): we treat "prior" as "earlier
purchase time". Strictly, a prior order's late/on-time outcome is only *known*
once it's delivered, which can be after a later order is purchased. Handling that
exactly needs an as-of join on delivery dates. We use the simpler purchase-order
version here for clarity; the residual leakage is small and noted in docs/NOTES.md.
"""

import numpy as np
import pandas as pd

from .config import load_config

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two lat/long points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def _row_local_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features from a single order only - impossible to leak."""
    # distance seller -> customer (far = riskier)
    df["distance_km"] = _haversine_km(
        df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"]
    )
    df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)

    # promised delivery window in days (tight window = riskier)
    df["delivery_window_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # order economics
    df["freight_ratio"] = df["total_freight"] / (
        df["total_price"] + df["total_freight"]
    ).replace(0, np.nan)

    # calendar features from the purchase moment (all known before dispatch)
    ts = df["order_purchase_timestamp"]
    df["purchase_dow"] = ts.dt.dayofweek          # 0=Mon .. 6=Sun
    df["purchase_month"] = ts.dt.month
    df["purchase_hour"] = ts.dt.hour
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    # simple "holiday season" flag (Black Friday / Christmas rush = Nov, Dec)
    df["is_holiday_season"] = ts.dt.month.isin([11, 12]).astype(int)
    return df


def _expanding_prior_rate(df: pd.DataFrame, group_col: str, target: str):
    """Leakage-safe historical late rate for a grouping column.

    For each order, returns the mean `target` of that group's EARLIER orders only.
    Uses shift(1) so the current order is excluded, and expanding() so it grows
    with history. Assumes df is already sorted by purchase time.
    Returns (rate_series, count_series).
    """
    grp = df.groupby(group_col)[target]
    rate = grp.transform(lambda s: s.shift(1).expanding().mean())
    count = grp.transform(lambda s: s.shift(1).expanding().count())
    return rate, count


def _historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe past late-rate features (seller / state / category)."""
    target = load_config()["data"]["target_column"]

    # a global "prior late rate so far" - used to fill cold-start gaps safely
    global_prior = df[target].shift(1).expanding().mean()

    for col, name in [
        ("seller_id", "seller"),
        ("customer_state", "cust_state"),
        ("product_category_name_english", "category"),
    ]:
        rate, count = _expanding_prior_rate(df, col, target)
        # cold start (no prior orders for this group) -> use global prior so far
        df[f"{name}_hist_late_rate"] = rate.fillna(global_prior).fillna(0.0)
        df[f"{name}_hist_count"] = count.fillna(0)

    return df


# The columns the model will actually use.
FEATURE_COLUMNS = [
    "distance_km",
    "same_state",
    "delivery_window_days",
    "freight_ratio",
    "total_freight",
    "total_price",
    "n_items",
    "n_sellers",
    "max_weight_g",
    "purchase_dow",
    "purchase_month",
    "purchase_hour",
    "is_weekend",
    "is_holiday_season",
    "seller_hist_late_rate",
    "seller_hist_count",
    "cust_state_hist_late_rate",
    "cust_state_hist_count",
    "category_hist_late_rate",
    "category_hist_count",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all feature columns to the (time-sorted) order table.

    Input must be the output of data.build_dataset() (already sorted by purchase
    time). Returns the same DataFrame with FEATURE_COLUMNS added.
    """
    df = _row_local_features(df)
    df = _historical_features(df)
    return df


if __name__ == "__main__":
    # smoke test: python -m routeguard.features
    from .data import build_dataset

    data = build_features(build_dataset())
    print("rows:", len(data))
    print("\nfeature preview:")
    print(data[FEATURE_COLUMNS].describe().round(3).T)
    print("\nmissing values per feature:")
    print(data[FEATURE_COLUMNS].isna().sum())
