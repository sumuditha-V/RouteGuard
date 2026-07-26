"""Load the Olist CSVs, join them, and build one row per order with the target.

This is the foundation everything else builds on. The output is a single pandas
DataFrame at ORDER granularity (one row = one order) with:
  - the target column `late_delivery` (1 = late, 0 = on time)
  - raw columns needed later for feature engineering (dates, freight, weight,
    seller/customer zip prefixes, geolocation lat/long, category)

We do NOT engineer risk features here (distance, historical rates, etc.) -
that happens in features.py so the leakage-safe logic stays in one place.
"""

import pandas as pd

from .config import load_config, raw_dir

# Columns in the orders table that are dates.
DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def _load_raw() -> dict[str, pd.DataFrame]:
    """Read the raw Olist CSVs we need into a dict of DataFrames."""
    rd = raw_dir()
    return {
        "orders": pd.read_csv(rd / "olist_orders_dataset.csv"),
        "items": pd.read_csv(rd / "olist_order_items_dataset.csv"),
        "products": pd.read_csv(rd / "olist_products_dataset.csv"),
        "sellers": pd.read_csv(rd / "olist_sellers_dataset.csv"),
        "customers": pd.read_csv(rd / "olist_customers_dataset.csv"),
        "geo": pd.read_csv(rd / "olist_geolocation_dataset.csv"),
        "cat": pd.read_csv(rd / "product_category_name_translation.csv"),
    }


def _geo_lookup(geo: pd.DataFrame) -> pd.DataFrame:
    """One representative lat/long per zip-code prefix (median, to reduce noise)."""
    return (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))
        .reset_index()
    )


def _items_per_order(items: pd.DataFrame, products: pd.DataFrame,
                     cat: pd.DataFrame) -> pd.DataFrame:
    """Collapse the (multi-row) order_items table down to one row per order.

    An order can contain several items from several sellers. We aggregate to
    order level and keep the FIRST item's seller/category as the 'primary' one.
    """
    # attach product weight + category to each item
    products = products.merge(cat, on="product_category_name", how="left")
    items = items.merge(
        products[["product_id", "product_weight_g", "product_category_name_english"]],
        on="product_id",
        how="left",
    )

    # per-order aggregates
    agg = (
        items.groupby("order_id")
        .agg(
            total_freight=("freight_value", "sum"),
            total_price=("price", "sum"),
            n_items=("order_item_id", "count"),
            n_sellers=("seller_id", "nunique"),
            max_weight_g=("product_weight_g", "max"),
        )
        .reset_index()
    )

    # primary seller + category = from the first item line of each order
    first_item = items.sort_values("order_item_id").groupby("order_id").first()
    agg = agg.merge(
        first_item[["seller_id", "product_category_name_english"]].reset_index(),
        on="order_id",
        how="left",
    )
    return agg


def build_dataset() -> pd.DataFrame:
    """Return one row per delivered order, with the target and base columns."""
    cfg = load_config()
    raw = _load_raw()
    orders = raw["orders"].copy()

    # parse dates
    for c in DATE_COLS:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    # keep only delivered orders that have both actual + estimated delivery dates
    status = cfg["data"]["order_status_filter"]
    df = orders[orders["order_status"] == status].copy()
    df = df.dropna(subset=["order_delivered_customer_date",
                           "order_estimated_delivery_date"])

    # TARGET: late if actual delivery is after the estimate
    df[cfg["data"]["target_column"]] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)

    # order-level item aggregates (freight, weight, seller, category, ...)
    df = df.merge(_items_per_order(raw["items"], raw["products"], raw["cat"]),
                  on="order_id", how="left")

    # customer + seller zip prefixes
    df = df.merge(
        raw["customers"][["customer_id", "customer_zip_code_prefix", "customer_state"]],
        on="customer_id", how="left",
    )
    df = df.merge(
        raw["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]],
        on="seller_id", how="left",
    )

    # attach lat/long for both ends (for Haversine distance later)
    geo = _geo_lookup(raw["geo"])
    df = df.merge(
        geo.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                            "lat": "customer_lat", "lng": "customer_lng"}),
        on="customer_zip_code_prefix", how="left",
    )
    df = df.merge(
        geo.rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix",
                            "lat": "seller_lat", "lng": "seller_lng"}),
        on="seller_zip_code_prefix", how="left",
    )

    # sort by purchase time - REQUIRED before any leakage-safe feature work
    df = df.sort_values(cfg["split"]["time_column"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # quick smoke test: python -m routeguard.data
    data = build_dataset()
    print("rows:", len(data))
    print("late rate:", round(data["late_delivery"].mean(), 4))
    print("columns:", list(data.columns))
