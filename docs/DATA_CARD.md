# Data Card — RouteGuard

## Source
**Brazilian E-Commerce Public Dataset by Olist** (Kaggle).
~100k orders, 2016–2018. Multiple related CSV tables joined on IDs.

Download the CSVs into `data/raw/` (gitignored). Record the exact Kaggle version/date here
once downloaded: _TODO_.

## Tables used
- `olist_orders_dataset` — order timestamps (purchase, approved, carrier handoff, delivered, estimated).
- `olist_order_items_dataset` — freight_value, price, shipping_limit_date, seller_id.
- `olist_products_dataset` — weight, dimensions, category.
- `olist_sellers_dataset`, `olist_customers_dataset` — zip code prefixes.
- `olist_geolocation_dataset` — lat/long per zip prefix (for Haversine distance).
- `product_category_name_translation` — English category names.

## Target
```
late_delivery = 1 if order_delivered_customer_date > order_estimated_delivery_date else 0
```
Only on `order_status == 'delivered'` rows with a non-null actual delivery date.
Base rate ~6–8% late (confirm during EDA).

## Known caveats (important)
1. **No carrier field.** Olist has a carrier *handoff date* but not carrier *identity*.
   → We **drop carrier as a real feature.** The agent action `choose_alternative_carrier`
     is kept only as an **illustrative** recommendation and is labeled as such in the UI.
2. **Geolocation is by zip *prefix*** (many-to-one, slightly noisy).
   → Use the **median** lat/long per prefix as the representative centroid.
3. **Pre-dispatch realism.** At prediction time we may only use features known **at/before
   purchase**. Carrier handoff date, actual delivery date, and approval time are excluded
   as leakage for a pre-dispatch model.

## Leakage discipline
Historical rate features (seller/region/category late rates) are computed with expanding
windows + `shift(1)`, and all transformers are fit on the training slice only. A unit test
enforces this.
