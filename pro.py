import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats

# Page config
st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide")

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
# 📌 Show Dataset
# -----------------------------
st.subheader("📂 Dataset")
st.dataframe(sales_data)

# -----------------------------
# 📊 Statistics
# -----------------------------
st.subheader("📈 Descriptive Statistics")

mean_sales = sales_data['units_sold'].mean()
median_sales = sales_data['units_sold'].median()
mode_sales = sales_data['units_sold'].mode()[0]
std_dev = sales_data['units_sold'].std()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Mean", round(mean_sales, 2))
col2.metric("Median", median_sales)
col3.metric("Mode", mode_sales)
col4.metric("Std Dev", round(std_dev, 2))

st.write(sales_data['units_sold'].describe())

# -----------------------------
# 📊 Category Analysis
# -----------------------------
st.subheader("📦 Category Analysis")

category_stats = sales_data.groupby('category')['units_sold'].agg(['sum', 'mean', 'std']).reset_index()
category_stats.columns = ['Category', 'Total Units Sold', 'Average Units Sold', 'Std Dev']

st.dataframe(category_stats)

# -----------------------------
# 🎯 Confidence Interval
# -----------------------------
st.subheader("🎯 Confidence Interval (95%)")

confidence_level = 0.95
df = len(sales_data['units_sold']) - 1
sample_mean = mean_sales
sample_std_error = std_dev / np.sqrt(len(sales_data))

t_score = stats.t.ppf((1 + confidence_level) / 2, df)
margin_error = t_score * sample_std_error

ci_lower = sample_mean - margin_error
ci_upper = sample_mean + margin_error

st.success(f"Confidence Interval: ({round(ci_lower,2)}, {round(ci_upper,2)})")

# -----------------------------
# 🧪 Hypothesis Testing
# -----------------------------
st.subheader("🧪 Hypothesis Testing (t-test)")

t_stat, p_value = stats.ttest_1samp(sales_data['units_sold'], 20)

st.write(f"T-statistic: {round(t_stat,4)}")
st.write(f"P-value: {round(p_value,4)}")

if p_value < 0.05:
    st.error("Reject Null Hypothesis ❌ (Mean is significantly different from 20)")
else:
    st.success("Fail to Reject Null Hypothesis ✅")

# -----------------------------
# 🎛️ Filter (Interactive Feature)
# -----------------------------
st.sidebar.header("🔍 Filter Data")

selected_category = st.sidebar.selectbox(
    "Select Category",
    options=["All"] + list(sales_data['category'].unique())
)

if selected_category != "All":
    filtered_data = sales_data[sales_data['category'] == selected_category]
else:
    filtered_data = sales_data

# -----------------------------
# 📊 Visualizations
# -----------------------------
st.subheader("📊 Visualizations")

# Histogram
st.write("### Distribution of Units Sold")
fig1, ax1 = plt.subplots()
sns.histplot(filtered_data['units_sold'], bins=10, kde=True, ax=ax1)
st.pyplot(fig1)

# Boxplot
st.write("### Boxplot by Category")
fig2, ax2 = plt.subplots()
sns.boxplot(x='category', y='units_sold', data=filtered_data, ax=ax2)
st.pyplot(fig2)

# Bar Chart
st.write("### Total Units Sold by Category")
fig3, ax3 = plt.subplots()
sns.barplot(x='Category', y='Total Units Sold', data=category_stats, ax=ax3)
st.pyplot(fig3)

st.success("✅ App Running Successfully!")