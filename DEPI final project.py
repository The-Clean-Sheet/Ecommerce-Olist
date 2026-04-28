#!/usr/bin/env python
# coding: utf-8

# In[8]:


# =========================
# IMPORTS
# =========================
import pandas as pd
import os
import zipfile
from datetime import datetime

# =========================
# READ FILES
# =========================
path = r"E:\Mokhtar\Final project"

files = [
    'olist_customers_dataset.csv',
    'olist_order_items_dataset.csv',
    'olist_order_payments_dataset.csv',
    'olist_order_reviews_dataset.csv',
    'olist_orders_dataset.csv',
    'olist_products_dataset.csv',
    'olist_sellers_dataset.csv',
    'product_category_name_translation.csv'
]

data = {}
for file in files:
    name = file.replace('.csv', '')
    data[name] = pd.read_csv(os.path.join(path, file))

# =========================
# OUTPUT FOLDER
# =========================
output_folder = os.path.join(path, "output_cleaned")
os.makedirs(output_folder, exist_ok=True)

# =========================
# VALIDATION LOGGER
# =========================
validation = []

def log_validation(stage, df_name, df):
    validation.append({
        "Stage": stage,
        "Dataset": df_name,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing Values": int(df.isna().sum().sum()),
        "Duplicates": int(df.duplicated().sum())
    })

# =========================
# SAFE DATETIME
# =========================
def safe_to_datetime(df, col):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# =========================
# BEFORE CLEANING
# =========================
for name, df in data.items():
    log_validation("Before", name, df)

# =========================
# CLEANING FUNCTIONS
# =========================

def clean_customers(df):
    df = df.copy()
    df['customer_zip_code_prefix'] = df['customer_zip_code_prefix'].astype(str).str.zfill(5)
    return df


def clean_order_items(df):
    df = df.copy()
    df['order_item_id'] = df['order_item_id'].astype(str)
    df = safe_to_datetime(df, 'shipping_limit_date')
    return df


def clean_payments(df):
    df = df.copy()
    # 0 installments doesn't make sense — treat as 1
    df['payment_installments'] = df['payment_installments'].replace(0, 1)
    return df


def clean_sellers(df):
    df = df.copy()
    df['seller_zip_code_prefix'] = df['seller_zip_code_prefix'].astype(str).str.zfill(5)
    return df


def clean_reviews(df):
    df = df.copy()
    # Keep first occurrence of each review
    df = df.drop_duplicates(subset=['review_id'], keep='first')
    return df


def clean_orders(df):
    df = df.copy()

    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        df = safe_to_datetime(df, col)

    # Fill missing approval time with purchase time
    df['order_approved_at'] = df['order_approved_at'].fillna(df['order_purchase_timestamp'])

    # Fix 'unavailable' and 'created' statuses — treat as approved
    df.loc[df['order_status'].isin(['unavailable', 'created']), 'order_status'] = 'approved'

    # Fix 'delivered' orders that have no delivered_customer_date (data inconsistency)
    mask_delivered = df['order_status'] == 'delivered'
    mask_no_customer_date = df['order_delivered_customer_date'].isna()
    mask_has_carrier_date = df['order_delivered_carrier_date'].notna()

    # Has carrier date but no customer date => reclassify as 'shipped'
    df.loc[mask_delivered & mask_no_customer_date & mask_has_carrier_date, 'order_status'] = 'shipped'

    # Has neither date => reclassify as 'approved'
    df.loc[mask_delivered & mask_no_customer_date & ~mask_has_carrier_date, 'order_status'] = 'approved'

    return df


def clean_products(products, translations):
    products = products.copy()

    # Merge with English translations
    products = products.merge(translations, on='product_category_name', how='left')
    products.rename(columns={'product_category_name_english': 'product_category'}, inplace=True)

    # Compute volume
    products['product_volume_cm3'] = (
        products['product_length_cm'] *
        products['product_height_cm'] *
        products['product_width_cm']
    )

    return products


# =========================
# APPLY CLEANING
# =========================
data['olist_customers_dataset']      = clean_customers(data['olist_customers_dataset'])
data['olist_order_items_dataset']    = clean_order_items(data['olist_order_items_dataset'])
data['olist_order_payments_dataset'] = clean_payments(data['olist_order_payments_dataset'])
data['olist_sellers_dataset']        = clean_sellers(data['olist_sellers_dataset'])
data['olist_order_reviews_dataset']  = clean_reviews(data['olist_order_reviews_dataset'])
data['olist_orders_dataset']         = clean_orders(data['olist_orders_dataset'])
data['olist_products_dataset']       = clean_products(
    data['olist_products_dataset'],
    data['product_category_name_translation']
)

# =========================
# AFTER CLEANING — LOG VALIDATION
# =========================
for name, df in data.items():
    log_validation("After", name, df)

# =========================
# FEATURE ENGINEERING
# (done BEFORE capitalize_columns to keep column names consistent)
# =========================
orders = data['olist_orders_dataset']

# Is the order actually delivered?
orders['is_delivered'] = orders['order_delivered_customer_date'].notna()

# Delivery delay in days (positive = late, negative = early)
# Only calculated for delivered orders — others stay NaN (not 0!)
orders['delivery_delay_days'] = (
    orders['order_delivered_customer_date'] -
    orders['order_estimated_delivery_date']
).dt.days

# Boolean: was the order late?
orders['is_delayed'] = orders['delivery_delay_days'].gt(0)

# Processing time in hours (purchase -> approval)
orders['processing_hours'] = (
    orders['order_approved_at'] -
    orders['order_purchase_timestamp']
).dt.total_seconds() / 3600

# Actual shipping time in days (carrier pickup -> customer delivery)
# Only meaningful for delivered orders — NaN otherwise
orders['shipping_time_days'] = (
    orders['order_delivered_customer_date'] -
    orders['order_delivered_carrier_date']
).dt.days

data['olist_orders_dataset'] = orders

# =========================
# CAPITALIZE COLUMNS
# (done AFTER all merges and feature engineering)
# =========================
def capitalize_columns(df):
    df.columns = [col.capitalize() for col in df.columns]
    return df

for name in data:
    data[name] = capitalize_columns(data[name])

# =========================
# INSIGHTS
# =========================
orders = data['olist_orders_dataset']   # re-reference after capitalize

total_orders      = len(orders)
delivered_orders  = int(orders['Is_delivered'].sum())

# Delay stats — only for delivered orders
delivered_mask    = orders['Is_delivered']
delay_series      = orders.loc[delivered_mask, 'Delivery_delay_days']
delay_rate        = orders.loc[delivered_mask, 'Is_delayed'].mean() * 100
avg_delay         = delay_series.mean()
avg_early         = delay_series[delay_series < 0].mean()

print(f"Total Orders    : {total_orders:,}")
print(f"Delivered Orders: {delivered_orders:,}")
print(f"Delay Rate      : {delay_rate:.2f}%  (among delivered orders)")
print(f"Avg Delay       : {avg_delay:.2f} days")
print(f"Avg Early Arrival: {avg_early:.2f} days")

# =========================
# SAVE TO ZIP
# =========================
zip_path = os.path.join(output_folder, "final_output.zip")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for name, df in data.items():
        temp_path = os.path.join(output_folder, f"{name}_cleaned.csv")
        df.to_csv(temp_path, index=False)
        zipf.write(temp_path, arcname=f"{name}_cleaned.csv")
        os.remove(temp_path)

print(f"\nZIP saved at: {zip_path}")

# =========================
# PDF REPORT
# =========================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

pdf_path = os.path.join(output_folder, "EDA_Report.pdf")
doc = SimpleDocTemplate(pdf_path)
styles = getSampleStyleSheet()
elements = []

# Title
elements.append(Paragraph("Auto EDA Report - Olist Dataset", styles['Title']))
elements.append(Spacer(1, 8))
elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
elements.append(Spacer(1, 20))

# ---- Dataset Overview Table (Before vs After) ----
elements.append(Paragraph("Dataset Overview - Before Cleaning", styles['Heading2']))
elements.append(Spacer(1, 6))

before_data = [["Dataset", "Rows", "Columns", "Missing", "Duplicates"]]
for item in validation:
    if item["Stage"] == "Before":
        before_data.append([
            item["Dataset"], item["Rows"], item["Columns"],
            item["Missing Values"], item["Duplicates"]
        ])

before_table = Table(before_data, hAlign='LEFT')
before_table.setStyle(TableStyle([
    ('BACKGROUND',  (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR',   (0, 0), (-1, 0), colors.whitesmoke),
    ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('GRID',        (0, 0), (-1, -1), 0.5, colors.black),
    ('PADDING',     (0, 0), (-1, -1), 5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
]))
elements.append(before_table)
elements.append(Spacer(1, 16))

elements.append(Paragraph("Dataset Overview - After Cleaning", styles['Heading2']))
elements.append(Spacer(1, 6))

after_data = [["Dataset", "Rows", "Columns", "Missing", "Duplicates"]]
for item in validation:
    if item["Stage"] == "After":
        after_data.append([
            item["Dataset"], item["Rows"], item["Columns"],
            item["Missing Values"], item["Duplicates"]
        ])

after_table = Table(after_data, hAlign='LEFT')
after_table.setStyle(TableStyle([
    ('BACKGROUND',  (0, 0), (-1, 0), colors.darkblue),
    ('TEXTCOLOR',   (0, 0), (-1, 0), colors.whitesmoke),
    ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('GRID',        (0, 0), (-1, -1), 0.5, colors.black),
    ('PADDING',     (0, 0), (-1, -1), 5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
]))
elements.append(after_table)
elements.append(Spacer(1, 20))

# ---- Key Insights ----
elements.append(Paragraph("Key Insights", styles['Heading2']))
elements.append(Spacer(1, 8))

insights_text = (
    f"Total Orders: {total_orders:,}<br/>"
    f"Delivered Orders: {delivered_orders:,}<br/>"
    f"Delay Rate (delivered only): {delay_rate:.2f}%<br/>"
    f"Average Delay: {avg_delay:.2f} days<br/>"
    f"Average Early Arrival: {avg_early:.2f} days"
)
elements.append(Paragraph(insights_text, styles['Normal']))
elements.append(Spacer(1, 20))

# ---- Order Status Distribution ----
elements.append(Paragraph("Order Status Distribution", styles['Heading2']))
elements.append(Spacer(1, 8))

status_counts = orders['Order_status'].value_counts().reset_index()
status_counts.columns = ["Status", "Count"]

status_table_data = [["Status", "Count"]] + status_counts.values.tolist()
status_table = Table(status_table_data, hAlign='LEFT')
status_table.setStyle(TableStyle([
    ('BACKGROUND',  (0, 0), (-1, 0), colors.darkblue),
    ('TEXTCOLOR',   (0, 0), (-1, 0), colors.whitesmoke),
    ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('GRID',        (0, 0), (-1, -1), 0.5, colors.black),
    ('PADDING',     (0, 0), (-1, -1), 5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
]))
elements.append(status_table)

# ---- Build PDF ----
doc.build(elements)
print(f"EDA PDF saved at: {pdf_path}")
print("\nFULL PIPELINE COMPLETED SUCCESSFULLY")


# In[ ]:




