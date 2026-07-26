"""Leakage tests - the headline correctness guarantee of this project.

If any of these fail, the historical-rate features are peeking at the future and
the whole model's reported performance would be untrustworthy.
"""

import pandas as pd

from routeguard.features import _historical_features


def _toy_df(last_label: int) -> pd.DataFrame:
    """Four orders from the same seller/state/category, sorted by purchase time.
    Only the LAST order's outcome is controlled by `last_label`.
    """
    return pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]
            ),
            "seller_id": ["s1"] * 4,
            "customer_state": ["SP"] * 4,
            "product_category_name_english": ["toys"] * 4,
            "late_delivery": [1, 0, 1, last_label],
        }
    )


def test_future_outcome_does_not_leak_into_past():
    """Changing the last order's label must not change any earlier order's features."""
    a = _historical_features(_toy_df(last_label=0))
    b = _historical_features(_toy_df(last_label=1))

    # the first 3 rows must be identical regardless of the 4th row's outcome
    cols = ["seller_hist_late_rate", "cust_state_hist_late_rate",
            "category_hist_late_rate"]
    pd.testing.assert_frame_equal(a[cols].iloc[:3], b[cols].iloc[:3])


def test_history_uses_only_prior_orders():
    """The 4th order's seller rate = mean of the first 3 outcomes (1, 0, 1) = 0.667."""
    df = _historical_features(_toy_df(last_label=1))
    assert round(df["seller_hist_late_rate"].iloc[3], 3) == 0.667
    # and it must NOT equal the mean including itself (which would be 0.75)
    assert round(df["seller_hist_late_rate"].iloc[3], 3) != 0.75


def test_first_order_has_no_self_leak():
    """The very first order has no prior history, so its count must be 0."""
    df = _historical_features(_toy_df(last_label=1))
    assert df["seller_hist_count"].iloc[0] == 0
