import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats

st.set_page_config(page_title="Sales Analyzer", layout="wide")

st.title("📊 Sales Data Analyzer (Upload CSV / Excel)")

# -----------------------------
# 📂 Upload File
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

# -----------------------------
# 📥 Read File Safely
# -----------------------------
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                sales_data = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                sales_data = pd.read_csv(uploaded_file, encoding='latin1')

        elif uploaded_file.name.endswith(".xlsx"):
            sales_data = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

else:
    st.warning("⚠️ No file uploaded → Using sample dataset")

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
# 📂 Show Data
# -----------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(sales_data)

# -----------------------------
# ⚠️ Column Check
# -----------------------------
if 'units_sold' not in sales_data.columns:
    st.error("❌ Your file must contain 'units_sold' column")
    st.stop()

# -----------------------------
# 📈 Statistics
# -----------------------------
st.subheader("📈 Statistics")

mean_sales = sales_data['units_sold'].mean()
median_sales = sales_data['units_sold'].median()
mode_sales = sales_data['units_sold'].mode()[0]
std_dev = sales_data['units_sold'].std()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Mean", round(mean_sales, 2))
col2.metric("Median", median_sales)
col3.metric("Mode", mode_sales)
col4.metric("Std Dev", round(std_dev, 2))

# -----------------------------
# 📦 Category Analysis
# -----------------------------
if 'category' in sales_data.columns:
    st.subheader("📦 Category Analysis")

    category_stats = sales_data.groupby('category')['units_sold'].agg(['sum', 'mean', 'std']).reset_index()
    category_stats.columns = ['Category', 'Total Units Sold', 'Average Units Sold', 'Std Dev']

    st.dataframe(category_stats)
else:
    st.warning("⚠️ 'category' column not found")

# -----------------------------
# 📊 Visualizations
# -----------------------------
st.subheader("📊 Visualizations")

# Histogram
fig1, ax1 = plt.subplots()
sns.histplot(sales_data['units_sold'], bins=10, kde=True, ax=ax1)
ax1.set_title("Units Sold Distribution")
fig1.tight_layout()
st.pyplot(fig1, clear_figure=True)

# Boxplot
if 'category' in sales_data.columns:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='category', y='units_sold', data=sales_data, ax=ax2)
    ax2.set_title("Category vs Units Sold")
    fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True)

# -----------------------------
# 🎯 Confidence Interval
# -----------------------------
st.subheader("🎯 Confidence Interval")

confidence_level = 0.95
df = len(sales_data) - 1
std_error = std_dev / np.sqrt(len(sales_data))

t_score = stats.t.ppf((1 + confidence_level) / 2, df)
margin = t_score * std_error

ci_lower = mean_sales - margin
ci_upper = mean_sales + margin

st.success(f"95% CI: ({round(ci_lower,2)}, {round(ci_upper,2)})")

# -----------------------------
# 🧪 Hypothesis Testing
# -----------------------------
st.subheader("🧪 Hypothesis Testing")

t_stat, p_value = stats.ttest_1samp(sales_data['units_sold'], 20)

st.write(f"T-statistic: {round(t_stat,4)}")
st.write(f"P-value: {round(p_value,4)}")

if p_value < 0.05:
    st.error("Reject Null Hypothesis")
else:
    st.success("Fail to Reject Null Hypothesis")

st.success("✅ App is fully working with real data!")
