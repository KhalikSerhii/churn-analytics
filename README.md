# Churn Analytics Dashboard

Передбачення відтоку клієнтів e-commerce на основі LightGBM.

**Live demo:** https://churn-analytics-khalik.streamlit.app/
**Dataset** https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn/

## Стек
Python, LightGBM, SHAP, Plotly, Streamlit

## Результати моделі
- LightGBM ROC-AUC: 0.997
- Головний фактор churnu: Tenure (перші 3 місяці)
- Скарги подвоюють ризик відтоку

## Структура
- notebooks/ — EDA, feature engineering, моделювання
- src/ — збережена модель
- app/ — Streamlit дашборд
