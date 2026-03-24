import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats

# Page config
st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("📊 Sales Data Analysis Dashboard")

# Generate dataset
np.random.seed(42)

data = {
    'product_id': range(1, 21),
    'product_name': [f'Product {i}' for i in range(1, 21)],
    'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 20),
    'units_sold': np.random.poisson(lam=20, size=20),
    'sale_date': pd.date_range(start='2023-01-01', periods=20, freq='D')
}

sales_data = pd.DataFrame(data)

# -----------------------------
# 📂 Dataset
# -----------------------------
st.subheader("Dataset")
st.dataframe(sales_data)

# -----------------------------
# 📈 Stats
# -----------------------------
st.subheader("Statistics")

mean_sales = sales_data['units_sold'].mean()
median_sales = sales_data['units_sold'].median()
mode_sales = sales_data['units_sold'].mode()[0]
std_dev = sales_data['units_sold'].std()

st.write(f"Mean: {mean_sales}")
st.write(f"Median: {median_sales}")
st.write(f"Mode: {mode_sales}")
st.write(f"Standard Deviation: {std_dev}")

# -----------------------------
# 📦 Category Analysis
# -----------------------------
category_stats = sales_data.groupby('category')['units_sold'].agg(['sum', 'mean', 'std']).reset_index()
category_stats.columns = ['Category', 'Total Units Sold', 'Average Units Sold', 'Std Dev']

st.subheader("Category Stats")
st.dataframe(category_stats)

# -----------------------------
# 🎛️ Filter
# -----------------------------
st.sidebar.header("Filter")

selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + list(sales_data['category'].unique())
)

if selected_category != "All":
    filtered_data = sales_data[sales_data['category'] == selected_category]
else:
    filtered_data = sales_data

# Safety check
if filtered_data.empty:
    st.warning("No data available")
else:

    # -----------------------------
    # 📊 Histogram
    # -----------------------------
    st.subheader("Distribution of Units Sold")

    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_data['units_sold'], bins=10, kde=True, ax=ax1)
    ax1.set_title("Units Sold Distribution")
    st.pyplot(fig1)

    # -----------------------------
    # 📊 Boxplot
    # -----------------------------
    st.subheader("Boxplot by Category")

    fig2, ax2 = plt.subplots()
    sns.boxplot(x='category', y='units_sold', data=filtered_data, ax=ax2)
    ax2.set_title("Category vs Units Sold")
    st.pyplot(fig2)

    # -----------------------------
    # 📊 Bar Chart
    # -----------------------------
    st.subheader("Total Units Sold by Category")

    fig3, ax3 = plt.subplots()
    sns.barplot(x='Category', y='Total Units Sold', data=category_stats, ax=ax3)
    ax3.set_title("Total Sales per Category")
    st.pyplot(fig3)

# -----------------------------
# 🎯 Confidence Interval
# -----------------------------
st.subheader("Confidence Interval")

confidence_level = 0.95
df = len(sales_data) - 1
sample_mean = mean_sales
std_error = std_dev / np.sqrt(len(sales_data))

t_score = stats.t.ppf((1 + confidence_level) / 2, df)
margin = t_score * std_error

ci_lower = sample_mean - margin
ci_upper = sample_mean + margin

st.success(f"95% Confidence Interval: ({round(ci_lower,2)}, {round(ci_upper,2)})")

# -----------------------------
# 🧪 Hypothesis Testing
# -----------------------------
st.subheader("Hypothesis Testing")

t_stat, p_value = stats.ttest_1samp(sales_data['units_sold'], 20)

st.write(f"T-statistic: {t_stat}")
st.write(f"P-value: {p_value}")

if p_value < 0.05:
    st.error("Reject Null Hypothesis")
else:
    st.success("Fail to Reject Null Hypothesis")

st.success("✅ App Running Successfully!")
