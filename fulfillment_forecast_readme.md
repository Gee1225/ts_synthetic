# 📦 SKU-Level Fulfillment Volume Forecasting (Mock Project)

This project simulates a machine learning pipeline for forecasting **daily inbound and outbound SKU-level volumes** across a network of fulfillment centers. The solution uses synthetic data and is designed for interview preparation or prototyping demand forecasting pipelines.

---

## 🔍 Objective

Forecast SKU-level inbound and outbound volume to support:

- Labor planning
- Truckload and dock scheduling
- Inventory replenishment
- SKU allocation and exception monitoring

---

## 📁 Data Sources (Synthetic CSV Files)

| File                     | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| `volume_log.csv`         | Daily inbound/outbound volumes by `center_id` and `sku_id` |
| `calendar.csv`           | Calendar metadata (weekend, holiday, promotion flags)      |
| `product_catalog.csv`    | SKU category, storage type, and size                       |
| `weather.csv`            | Center-level weather alerts and temperature                |
| `carrier_tracking.csv`   | ETA vs. actual truck arrival, delay tracking               |
| `order_feed_stream.csv`  | Real-time order volume per SKU per center                  |
| `inventory_snapshot.csv` | Inventory availability, reorder point, lead time           |

---

## 🧠 Forecasting Logic (Simplified)

### Features:

- Lag features: last day, last week
- Rolling features: 7-day average, 14-day std deviation
- Calendar features: weekday, holiday, promo
- Weather: temperature, alert flag
- Order feed: last 24h quantity
- Carrier: delay minutes, late flag
- Inventory: available units, stockout risk, lead time

### Model:

- Baseline: SARIMAX (per SKU-center)
- Advanced: XGBoost regressor (all features above)
- Metrics: MAE, RMSE, WAPE

---

## 🚀 How to Use

1. **Run the Python script** to generate features and (placeholder) forecasts:

```bash
python forecast_pipeline.py
```

2. **Explore outputs**:

- Forecast performance per SKU/center/date
- Print sample feature matrix

---

## 🧪 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, matplotlib (optional)

---

## 📊 Next Steps (Not Included in Code Yet)

- Incorporate real-time streaming data
- Add visual dashboards
- Scale to multiple forecast horizons (7-day, 14-day)
- Add uncertainty estimates (prediction intervals)

---

## 🧬 License

For educational and prototyping use only. Synthetic data generated for demo purposes.

