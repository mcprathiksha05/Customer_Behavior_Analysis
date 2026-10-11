
from pathlib import Path
from io import BytesIO
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ============================================================
# CUSTOMER BEHAVIOR ANALYSIS - INTERNSHIP PROJECT
# ============================================================

st.set_page_config(
    page_title="Customer Behavior Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid")

# -------------------- PAGE DESIGN ---------------------------

st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #2563eb;
}
.subtitle {
    font-size: 16px;
    color: #64748b;
}
div[data-testid="stMetric"] {
    background-color: rgba(100, 116, 139, 0.08);
    border: 1px solid rgba(100, 116, 139, 0.18);
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📊 Customer Behavior Analysis</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">An interactive data science project for '
    'understanding customer purchasing patterns.</div>',
    unsafe_allow_html=True
)
st.write("")

# -------------------- SAMPLE DATA ---------------------------

@st.cache_data
def create_sample_data():
    """Create a reproducible demonstration dataset."""

    rng = np.random.default_rng(42)

    categories = [
        "Electronics", "Clothing", "Home & Kitchen",
        "Beauty", "Sports"
    ]

    products = {
        "Electronics": [
            "Headphones", "Smartwatch", "Power Bank", "Keyboard"
        ],
        "Clothing": [
            "T-shirt", "Jeans", "Jacket", "Sneakers"
        ],
        "Home & Kitchen": [
            "Bottle", "Cookware", "Lamp", "Storage Box"
        ],
        "Beauty": [
            "Face Wash", "Moisturizer", "Sunscreen", "Shampoo"
        ],
        "Sports": [
            "Yoga Mat", "Dumbbells", "Water Bottle", "Resistance Bands"
        ]
    }

    customers = [f"CUST{i:04d}" for i in range(1, 151)]
    regions = ["North", "South", "East", "West"]
    channels = ["Online", "In-store", "Mobile App"]

    dates = pd.to_datetime(
        rng.choice(
            pd.date_range("2025-01-01", "2026-09-30"),
            size=600
        )
    )

    rows = []

    for i in range(600):
        category = rng.choice(
            categories,
            p=[0.24, 0.25, 0.20, 0.16, 0.15]
        )

        quantity = int(rng.integers(1, 5))

        if category == "Electronics":
            price = round(float(rng.uniform(8, 220)), 2)
        else:
            price = round(float(rng.uniform(5, 90)), 2)

        rows.append({
            "Order_ID": f"ORD{i + 1:05d}",
            "Customer_ID": rng.choice(customers),
            "Order_Date": dates[i].strftime("%Y-%m-%d"),
            "Product": rng.choice(products[category]),
            "Category": category,
            "Quantity": quantity,
            "Unit_Price": price,
            "Region": rng.choice(regions),
            "Purchase_Channel": rng.choice(
                channels, p=[0.48, 0.28, 0.24]
            ),
            "Satisfaction_Score": int(rng.integers(1, 6))
        })

    data = pd.DataFrame(rows)

    # Intentionally add data-quality issues for demonstration.
    data.loc[[7, 41, 103], "Region"] = None
    data.loc[[18, 199], "Satisfaction_Score"] = np.nan

    # Add duplicate rows to demonstrate duplicate removal.
    data = pd.concat(
        [data, data.iloc[[10, 25]]],
        ignore_index=True
    )

    return data


# -------------------- DATA LOADING --------------------------

st.sidebar.title("⚙️ Project Controls")
st.sidebar.write("Upload your own customer dataset or use the sample data.")

uploaded_file = st.sidebar.file_uploader(
    "Upload customer CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
        data_source = "Uploaded dataset"
    except Exception as error:
        st.error(f"Could not read your CSV file: {error}")
        st.stop()
else:
    raw_data = create_sample_data()
    data_source = "Built-in synthetic sample dataset"

st.caption(f"Data source: {data_source}")

# -------------------- DATA VALIDATION -----------------------

required_columns = [
    "Order_ID",
    "Customer_ID",
    "Order_Date",
    "Product",
    "Category",
    "Quantity",
    "Unit_Price",
    "Region",
    "Purchase_Channel"
]

missing_columns = [
    col for col in required_columns
    if col not in raw_data.columns
]

if missing_columns:
    st.error(
        "Your CSV is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.info(
        "Required columns: " + ", ".join(required_columns)
    )
    st.stop()

# -------------------- DATA CLEANING -------------------------

data = raw_data.copy()

original_rows = len(data)
original_duplicates = int(data.duplicated().sum())
original_missing = int(data.isna().sum().sum())

# Remove exact duplicate rows.
data = data.drop_duplicates().copy()

# Convert date and numeric columns.
data["Order_Date"] = pd.to_datetime(
    data["Order_Date"], errors="coerce"
)

data["Quantity"] = pd.to_numeric(
    data["Quantity"], errors="coerce"
)

data["Unit_Price"] = pd.to_numeric(
    data["Unit_Price"], errors="coerce"
)

# Remove rows without essential fields.
data = data.dropna(
    subset=[
        "Order_ID",
        "Customer_ID",
        "Order_Date",
        "Quantity",
        "Unit_Price"
    ]
).copy()

# Remove invalid quantities and negative prices.
data = data[
    (data["Quantity"] > 0)
    & (data["Unit_Price"] >= 0)
].copy()

# Clean text columns.
text_columns = [
    "Order_ID",
    "Customer_ID",
    "Product",
    "Category",
    "Region",
    "Purchase_Channel"
]

for col in text_columns:
    data[col] = (
        data[col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

# Clean optional satisfaction score.
if "Satisfaction_Score" in data.columns:
    data["Satisfaction_Score"] = pd.to_numeric(
        data["Satisfaction_Score"], errors="coerce"
    )

    data.loc[
        ~data["Satisfaction_Score"].between(1, 5),
        "Satisfaction_Score"
    ] = np.nan

    median_score = data["Satisfaction_Score"].median()

    if pd.isna(median_score):
        median_score = 3.0

    data["Satisfaction_Score"] = (
        data["Satisfaction_Score"].fillna(median_score)
    )

# Calculate sales and date-based features.
data["Sales"] = data["Quantity"] * data["Unit_Price"]
data["Month"] = data["Order_Date"].dt.to_period("M").astype(str)
data["Year"] = data["Order_Date"].dt.year

cleaned_rows = len(data)
removed_rows = original_rows - cleaned_rows

if data.empty:
    st.error("No valid records remain after cleaning your dataset.")
    st.stop()

# -------------------- FILTERS ------------------------------

st.sidebar.subheader("🔎 Filter Analysis")

minimum_date = data["Order_Date"].min().date()
maximum_date = data["Order_Date"].max().date()

date_range = st.sidebar.date_input(
    "Order date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date
)

filtered_data = data.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

    filtered_data = filtered_data[
        filtered_data["Order_Date"].between(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date) + pd.Timedelta(days=1)
            - pd.Timedelta(microseconds=1)
        )
    ]

for column in ["Region", "Category", "Purchase_Channel"]:
    available = sorted(
        filtered_data[column].dropna().unique().tolist()
    )

    selected = st.sidebar.multiselect(
        f"Select {column.replace('_', ' ')}",
        options=available,
        default=available
    )

    filtered_data = filtered_data[
        filtered_data[column].isin(selected)
    ]

if filtered_data.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# -------------------- KPI CALCULATIONS ---------------------

total_sales = filtered_data["Sales"].sum()
total_orders = filtered_data["Order_ID"].nunique()
total_customers = filtered_data["Customer_ID"].nunique()
total_units = filtered_data["Quantity"].sum()

average_order_value = (
    total_sales / total_orders if total_orders else 0
)

# -------------------- KPI DASHBOARD ------------------------

st.subheader("📌 Business Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
col2.metric("🛒 Total Orders", f"{total_orders:,}")
col3.metric("👥 Customers", f"{total_customers:,}")
col4.metric("📦 Units Sold", f"{total_units:,.0f}")
col5.metric("🧾 Average Order Value", f"${average_order_value:,.2f}")

st.divider()

# -------------------- DASHBOARD TABS -----------------------

overview_tab, customer_tab, product_tab, quality_tab, report_tab = (
    st.tabs([
        "📈 Sales Overview",
        "👥 Customer Insights",
        "🛍️ Product Analysis",
        "🧹 Data Cleaning",
        "📄 Reports & Downloads"
    ])
)

# ============================================================
# TAB 1: SALES OVERVIEW
# ============================================================

with overview_tab:

    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        filtered_data.groupby("Month", as_index=False)["Sales"]
        .sum()
        .sort_values("Month")
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        monthly_sales["Month"],
        monthly_sales["Sales"],
        marker="o",
        linewidth=2
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Monthly Revenue Trend")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Sales by Category")

        category_sales = (
            filtered_data.groupby("Category", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=category_sales,
            x="Sales",
            y="Category",
            ax=ax
        )

        ax.set_title("Revenue by Product Category")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Sales by Purchase Channel")

        channel_sales = (
            filtered_data.groupby("Purchase_Channel")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.pie(
            channel_sales.values,
            labels=channel_sales.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Revenue Share by Channel")
        ax.axis("equal")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Orders by Region")

    region_orders = (
        filtered_data.groupby("Region")["Order_ID"]
        .nunique()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        x=region_orders.index,
        y=region_orders.values,
        ax=ax
    )

    ax.set_xlabel("Region")
    ax.set_ylabel("Unique Orders")
    ax.set_title("Order Distribution by Region")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Recent Transactions")

    st.dataframe(
        filtered_data.sort_values(
            "Order_Date", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# TAB 2: CUSTOMER INSIGHTS
# ============================================================

with customer_tab:

    st.subheader("Customer Spending and Purchase Frequency")

    customer_summary = (
        filtered_data.groupby("Customer_ID")
        .agg(
            Orders=("Order_ID", "nunique"),
            Total_Spend=("Sales", "sum"),
            Total_Units=("Quantity", "sum"),
            Last_Order=("Order_Date", "max")
        )
        .reset_index()
    )

    customer_summary["Average_Order_Value"] = (
        customer_summary["Total_Spend"]
        / customer_summary["Orders"].clip(lower=1)
    )

    # Segment customers using spending terciles.
    low_threshold = customer_summary["Total_Spend"].quantile(1 / 3)
    high_threshold = customer_summary["Total_Spend"].quantile(2 / 3)

    customer_summary["Segment"] = np.select(
        [
            customer_summary["Total_Spend"] <= low_threshold,
            customer_summary["Total_Spend"] <= high_threshold
        ],
        [
            "Low Spend",
            "Medium Spend"
        ],
        default="High Spend"
    )

    segment_summary = (
        customer_summary.groupby("Segment")
        .agg(
            Customers=("Customer_ID", "nunique"),
            Revenue=("Total_Spend", "sum")
        )
        .reindex(
            ["Low Spend", "Medium Spend", "High Spend"],
            fill_value=0
        )
        .reset_index()
    )

    left, right = st.columns(2)

    with left:
        st.write("Customer Count by Spending Segment")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.barplot(
            data=segment_summary,
            x="Segment",
            y="Customers",
            order=["Low Spend", "Medium Spend", "High Spend"],
            ax=ax
        )

        ax.set_xlabel("Customer Segment")
        ax.set_ylabel("Number of Customers")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.write("Purchase Frequency vs Total Spending")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.scatterplot(
            data=customer_summary,
            x="Orders",
            y="Total_Spend",
            hue="Segment",
            ax=ax
        )

        ax.set_xlabel("Number of Orders")
        ax.set_ylabel("Total Spending")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Customer Summary Table")

    st.dataframe(
        customer_summary.sort_values(
            "Total_Spend", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Customer Summary",
        data=customer_summary.to_csv(index=False).encode("utf-8"),
        file_name="customer_summary.csv",
        mime="text/csv"
    )

    if "Satisfaction_Score" in filtered_data.columns:
        average_satisfaction = (
            filtered_data["Satisfaction_Score"].mean()
        )

        st.metric(
            "⭐ Average Customer Satisfaction",
            f"{average_satisfaction:.2f} / 5"
        )

        satisfaction_by_category = (
            filtered_data.groupby("Category")["Satisfaction_Score"]
            .mean()
            .sort_values(ascending=False)
        )

        st.write("Satisfaction by Category")

        st.dataframe(
            satisfaction_by_category.rename(
                "Average Satisfaction"
            ).reset_index(),
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# TAB 3: PRODUCT ANALYSIS
# ============================================================

with product_tab:

    st.subheader("Product Performance")

    product_summary = (
        filtered_data.groupby(["Product", "Category"])
        .agg(
            Units_Sold=("Quantity", "sum"),
            Orders=("Order_ID", "nunique"),
            Revenue=("Sales", "sum")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )

    left, right = st.columns(2)

    with left:
        st.write("Top 10 Products by Revenue")

        top_products = product_summary.head(10).sort_values(
            "Revenue", ascending=True
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=top_products,
            x="Revenue",
            y="Product",
            ax=ax
        )

        ax.set_xlabel("Revenue")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.write("Top 10 Products by Units Sold")

        top_units = product_summary.sort_values(
            "Units_Sold", ascending=False
        ).head(10).sort_values("Units_Sold")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=top_units,
            x="Units_Sold",
            y="Product",
            ax=ax
        )

        ax.set_xlabel("Units Sold")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Complete Product Performance Table")

    st.dataframe(
        product_summary,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# TAB 4: DATA CLEANING
# ============================================================

with quality_tab:

    st.subheader("Data Quality Summary")

    quality_summary = pd.DataFrame({
        "Metric": [
            "Original row count",
            "Duplicate rows removed",
            "Rows removed during cleaning",
            "Final cleaned row count",
            "Missing cells in original data",
            "Missing cells in cleaned data"
        ],
        "Value": [
            original_rows,
            original_duplicates,
            removed_rows,
            cleaned_rows,
            original_missing,
            int(data.isna().sum().sum())
        ]
    })

    st.dataframe(
        quality_summary,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Missing Values in Original Data")

    missing_summary = (
        raw_data.isna().sum()
        .sort_values(ascending=False)
        .rename("Missing Values")
        .reset_index()
    )

    missing_summary.columns = ["Column", "Missing Values"]

    st.dataframe(
        missing_summary,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Data Cleaning Steps Performed")

    st.markdown("""
    1. Removed duplicate records.
    2. Converted order dates into date format.
    3. Converted quantity and unit price into numeric values.
    4. Removed records missing essential identifiers, dates,
       quantities, or prices.
    5. Removed invalid quantities and negative prices.
    6. Replaced missing descriptive text values with `Unknown`.
    7. Replaced missing satisfaction scores with the median.
    8. Calculated sales using quantity multiplied by unit price.
    9. Created month and year columns for trend analysis.
    """)

    st.subheader("Cleaned Dataset Preview")

    st.dataframe(
        data.head(50),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# TAB 5: REPORTS AND DOWNLOADS
# ============================================================

with report_tab:

    st.subheader("Business Insights")

    category_totals = (
        filtered_data.groupby("Category")["Sales"].sum()
    )

    region_totals = (
        filtered_data.groupby("Region")["Sales"].sum()
    )

    channel_totals = (
        filtered_data.groupby("Purchase_Channel")["Sales"].sum()
    )

    product_totals = (
        filtered_data.groupby("Product")["Sales"].sum()
    )

    insights = [
        f"Total revenue is ${total_sales:,.2f}.",
        f"There are {total_orders:,} unique orders.",
        f"There are {total_customers:,} unique customers.",
        f"Average order value is ${average_order_value:,.2f}."
    ]

    if not category_totals.empty:
        insights.append(
            f"Highest-revenue category: "
            f"{category_totals.idxmax()}."
        )

    if not region_totals.empty:
        insights.append(
            f"Highest-revenue region: "
            f"{region_totals.idxmax()}."
        )

    if not channel_totals.empty:
        insights.append(
            f"Highest-revenue purchase channel: "
            f"{channel_totals.idxmax()}."
        )

    if not product_totals.empty:
        insights.append(
            f"Highest-revenue product: "
            f"{product_totals.idxmax()}."
        )

    for i, insight in enumerate(insights, start=1):
        st.markdown(f"**{i}.** {insight}")

    st.subheader("Business Recommendations")

    st.markdown("""
    - Introduce loyalty benefits for high-spending customers.
    - Offer relevant promotions to encourage repeat purchases.
    - Review low-performing product categories and sales channels.
    - Track satisfaction scores to identify customer experience issues.
    - Monitor monthly sales trends to support inventory planning.
    - Repeat the analysis regularly to identify changing customer behavior.
    """)

    # Build report from the current filtered data.
    report_text = f"""# Customer Behavior Analysis Report

## 1. Project Overview

This project analyzes customer transactions using Python data science
libraries and an interactive Streamlit dashboard.

## 2. Dataset

- Data source: {data_source}
- Original rows: {original_rows}
- Duplicate rows removed: {original_duplicates}
- Cleaned rows: {cleaned_rows}
- Current filtered rows: {len(filtered_data)}

The built-in sample dataset is synthetic and does not represent
actual customer transactions.

## 3. Key Performance Indicators

- Total sales: ${total_sales:,.2f}
- Unique orders: {total_orders}
- Unique customers: {total_customers}
- Units sold: {total_units:,.0f}
- Average order value: ${average_order_value:,.2f}

## 4. Business Insights

""" + "\n".join(
        f"- {item}" for item in insights
    ) + """

## 5. Methodology

1. Load the customer transaction data.
2. Remove duplicates and invalid records.
3. Handle missing values and convert data types.
4. Calculate sales using quantity multiplied by unit price.
5. Analyze monthly, category, regional, and channel performance.
6. Calculate customer spending and purchase frequency.
7. Visualize results and prepare business recommendations.

## 6. Recommendations

- Create loyalty programs for high-value customers.
- Encourage repeat purchases with relevant promotions.
- Review category and channel performance before reallocating budgets.
- Monitor customer satisfaction and monthly revenue.

## 7. Limitations

Results depend on the selected dataset and filters. The included
sample dataset is synthetic. Use authorized real data for genuine
business conclusions.

## 8. Conclusion

This project demonstrates a practical workflow for data cleaning,
customer behavior analysis, visualization, and report generation.
"""

    st.download_button(
        "📄 Download Project Report",
        data=report_text.encode("utf-8"),
        file_name="customer_behavior_report.md",
        mime="text/markdown"
    )

    st.download_button(
        "📥 Download Cleaned Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_customer_orders.csv",
        mime="text/csv"
    )

    st.download_button(
        "📥 Download Filtered Analysis Data",
        data=filtered_data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_analysis.csv",
        mime="text/csv"
    )

    st.success(
        "Your report and datasets are ready to download."
    )

# -------------------- FOOTER --------------------------------

st.divider()

st.caption(
    "Customer Behavior Analysis | Python | Pandas | NumPy | "
    "Matplotlib | Seaborn | Streamlit"
)
