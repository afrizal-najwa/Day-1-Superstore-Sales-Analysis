# =========================
# Day 1 Mini-Dashboard: Superstore Sales (Kaggle)
# Author: Afrizal Najwa Syauqi
# =========================
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Day 1 — Superstore Sales Analytics", page_icon="📊", layout="wide")

# ---------- THEME ----------
PRIMARY = "#2563eb"
ACCENT = "#10b981"
WARNING = "#f59e0b"
DANGER  = "#ef4444"
NEUTRAL = "#64748b"

HIDE_STYLES = """
<style>
/* compact paddings */
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
/* card look */
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""
st.markdown(HIDE_STYLES, unsafe_allow_html=True)

# ---------- HELPERS ----------
@st.cache_data(show_spinner=False)
def load_csv_safely(path_or_buffer):
    """
    Try UTF-8 -> latin-1 -> ISO-8859-1
    Return df, encoding_used
    """
    for enc in ["utf-8", "latin-1", "ISO-8859-1"]:
        try:
            df = pd.read_csv(path_or_buffer, encoding=enc)
            return df, enc
        except UnicodeDecodeError:
            continue
    # last resort: read as binary then decode non-breaking space
    if isinstance(path_or_buffer, str):
        with open(path_or_buffer, "rb") as f:
            raw = f.read()
    else:
        raw = path_or_buffer.read()
    fixed = raw.replace(b"\xa0", b" ")
    df = pd.read_csv(io.BytesIO(fixed))
    return df, "binary-fix"

# ---------- SAFE DATAFRAME (fallback tanpa pyarrow) ----------
def safe_dataframe(df, **kwargs):
    """
    Coba pakai st.dataframe (butuh pyarrow). Kalau ImportError/ DLL error,
    render sebagai HTML table agar tetap tampil.
    """
    try:
        import pyarrow  # noqa: F401 - sekadar test import
        return st.dataframe(df, **kwargs)
    except Exception as e:
        st.warning(
            "⚠️ PyArrow tidak bisa diload. Menampilkan tabel fallback (HTML). "
            f"Detail: {type(e).__name__}: {e}"
        )
        # Render HTML table rapi (tanpa index)
        try:
            html = df.to_html(index=False, border=0)
            st.markdown(
                f"""
                <div style="max-height:{kwargs.get('height',420)}px; overflow:auto; border:1px solid #e5e7eb; border-radius:8px; padding:8px;">
                    {html}
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as ee:
            st.error(f"Gagal render fallback HTML: {ee}")


def to_money(x):
    try:
        return f"${x:,.0f}"
    except Exception:
        return x

def normalize_columns(df: pd.DataFrame):
    # Standardize expected Kaggle Superstore columns
    cols = {c.strip().lower(): c for c in df.columns}
    # required base names in lower
    mapping_candidates = {
        "order date": ["order date", "order_date", "date"],
        "sales": ["sales", "revenue", "amount"],
        "profit": ["profit", "margin"],
        "quantity": ["quantity", "qty"],
        "category": ["category"],
        "sub-category": ["sub-category", "subcategory", "sub_category"],
        "region": ["region", "area"],
        "segment": ["segment", "customer segment", "cust_segment"]
    }
    final_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for std_name, alts in mapping_candidates.items():
        for a in alts:
            if a in lower_cols:
                final_map[lower_cols[a]] = std_name.title() if std_name != "sub-category" else "Sub-Category"
                break
    # apply
    df2 = df.rename(columns=final_map)
    return df2

def parse_dates(df: pd.DataFrame):
    # parse Order Date if exists
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        # derive Year-Month
        df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)
        df["Year"] = df["Order Date"].dt.year
        df["Month"] = df["Order Date"].dt.month
    return df

@st.cache_data(show_spinner=False)
def load_data(default_path="data/SampleSuperstore.csv", file_uploader_bytes=None):
    enc_used = None
    if file_uploader_bytes is not None:
        df, enc_used = load_csv_safely(file_uploader_bytes)
    else:
        if os.path.exists(default_path):
            df, enc_used = load_csv_safely(default_path)
        else:
            raise FileNotFoundError(
                "Tidak menemukan 'SampleSuperstore.csv'. Upload file lewat sidebar!"
            )
    df = normalize_columns(df)
    df = parse_dates(df)

    # Coerce numerics
    for col in ["Sales", "Profit", "Quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, enc_used

def apply_filters(df, date_range, regions, categories, segments):
    if "Order Date" in df.columns and date_range:
        start, end = date_range
        if pd.notnull(start) and pd.notnull(end):
            df = df[(df["Order Date"] >= pd.to_datetime(start)) & (df["Order Date"] <= pd.to_datetime(end))]
    if "Region" in df.columns and regions:
        df = df[df["Region"].isin(regions)]
    if "Category" in df.columns and categories:
        df = df[df["Category"].isin(categories)]
    if "Segment" in df.columns and segments:
        df = df[df["Segment"].isin(segments)]
    return df

# ---------- SIDEBAR ----------
st.sidebar.markdown("## 📥 Data Source")
uploaded = st.sidebar.file_uploader("Upload `SampleSuperstore.csv` (Kaggle)", type=["csv"])
st.sidebar.caption("Jika tidak upload, app akan mencoba membaca `SampleSuperstore.csv` di folder yang sama.")

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🔧 Options")

# Load data
try:
    df, encoding_used = load_data(file_uploader_bytes=uploaded if uploaded else None)
    st.sidebar.success(f"Loaded ✓  Encoding: `{encoding_used}`")
except Exception as e:
    st.sidebar.error(f"Gagal load data: {e}")
    st.stop()

# Filters
min_date = df["Order Date"].min() if "Order Date" in df.columns else None
max_date = df["Order Date"].max() if "Order Date" in df.columns else None

if min_date is not None and max_date is not None:
    date_range = st.sidebar.date_input("Tanggal (range)", value=(min_date.date(), max_date.date()))
else:
    date_range = None

regions = []
categories = []
segments = []

if "Region" in df.columns:
    regions = st.sidebar.multiselect("Filter Region", sorted(df["Region"].dropna().unique().tolist()))
if "Category" in df.columns:
    categories = st.sidebar.multiselect("Filter Category", sorted(df["Category"].dropna().unique().tolist()))
if "Segment" in df.columns:
    segments = st.sidebar.multiselect("Filter Segment", sorted(df["Segment"].dropna().unique().tolist()))

st.sidebar.markdown("---")
dl_fmt = st.sidebar.selectbox("Format download agregasi", ["csv", "xlsx"], index=0)

# Apply filters
dff = apply_filters(df.copy(), date_range, regions, categories, segments)

# ---------- HEADER ----------
st.title("📊 Day 1 — Superstore Sales Analytics (Kaggle)")
st.caption("Dataset: Kaggle Superstore — Analisis penjualan, profit, dan tren dasar untuk branding #100DaysOfAI")

# ---------- KPI ROW ----------
c1, c2, c3, c4 = st.columns(4)
total_sales = dff["Sales"].sum() if "Sales" in dff.columns else np.nan
total_profit = dff["Profit"].sum() if "Profit" in dff.columns else np.nan
total_orders = dff.shape[0]
aov = (total_sales / total_orders) if total_orders > 0 else np.nan

c1.metric("Total Sales", to_money(total_sales) if np.isfinite(total_sales) else "-")
c2.metric("Total Profit", to_money(total_profit) if np.isfinite(total_profit) else "-",
          delta=to_money(total_profit/total_sales*100) + "%" if (np.isfinite(total_sales) and total_sales>0) else None)
c3.metric("Total Orders", f"{total_orders:,}")
c4.metric("Avg Order Value (AOV)", to_money(aov) if np.isfinite(aov) else "-")

st.markdown("---")

# ---------- ROW: CATEGORY & SUBCATEGORY ----------
cc1, cc2 = st.columns([1,1])

with cc1:
    st.subheader("Sales by Category")
    if {"Category","Sales"}.issubset(dff.columns):
        cat_agg = dff.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
        fig_cat = px.bar(cat_agg, x="Category", y="Sales", text="Sales",
                         title=None)
        fig_cat.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_cat.update_layout(yaxis_title="Sales", xaxis_title="", margin=dict(t=10,b=10))
        st.plotly_chart(fig_cat, use_container_width=True)
        st.download_button(
            "download 'Sales by Category'",
            data=cat_agg.to_csv(index=False).encode("utf-8"),
            file_name="sales_by_category.csv",
            mime="text/csv",
        )
    else:
        st.info("Kolom 'Category' atau 'Sales' tidak tersedia.")

with cc2:
    st.subheader("Top 10 Sub-Categories by Sales")
    if {"Sub-Category","Sales"}.issubset(dff.columns):
        sub_agg = (
            dff.groupby("Sub-Category", as_index=False)["Sales"]
            .sum().sort_values("Sales", ascending=False).head(10)
        )
        fig_sub = px.bar(sub_agg, x="Sub-Category", y="Sales", text="Sales")
        fig_sub.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_sub.update_layout(yaxis_title="Sales", xaxis_title="", margin=dict(t=10,b=10))
        st.plotly_chart(fig_sub, use_container_width=True)
        # Download in chosen format
        if dl_fmt == "csv":
            st.download_button(
                "download 'Top Sub-Categories' (CSV)",
                data=sub_agg.to_csv(index=False).encode("utf-8"),
                file_name="top_subcategories.csv",
                mime="text/csv",
            )
        else:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                sub_agg.to_excel(writer, index=False, sheet_name="TopSubcategories")
            st.download_button(
                "download 'Top Sub-Categories' (XLSX)",
                data=buf.getvalue(),
                file_name="top_subcategories.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("Kolom 'Sub-Category' atau 'Sales' tidak tersedia.")

st.markdown("---")

# ---------- ROW: REGION & PROFITABILITY ----------
rc1, rc2 = st.columns([1,1])

with rc1:
    st.subheader("Profit by Region")
    if {"Region","Profit"}.issubset(dff.columns):
        reg_agg = (
            dff.groupby("Region", as_index=False)["Profit"]
            .sum().sort_values("Profit", ascending=False)
        )
        fig_reg = px.bar(reg_agg, x="Region", y="Profit", text="Profit",
                         color="Profit", color_continuous_scale="RdYlGn")
        fig_reg.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_reg.update_layout(yaxis_title="Profit", xaxis_title="", margin=dict(t=10,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("Kolom 'Region' atau 'Profit' tidak tersedia.")

with rc2:
    st.subheader("Profitability Heatmap (Category × Segment)")
    if {"Category","Segment","Profit"}.issubset(dff.columns):
        pivot = dff.pivot_table(index="Category", columns="Segment", values="Profit", aggfunc="sum", fill_value=0)
        pivot_reset = pivot.reset_index().melt(id_vars="Category", var_name="Segment", value_name="Profit")
        fig_heat = px.density_heatmap(
            pivot_reset, x="Segment", y="Category", z="Profit",
            color_continuous_scale="RdYlGn", nbinsx=len(pivot_reset["Segment"].unique())
        )
        fig_heat.update_layout(margin=dict(t=10,b=10))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Perlu 'Category', 'Segment', dan 'Profit'.")

st.markdown("---")

# ---------- TIME SERIES ----------
st.subheader("Monthly Sales Trend")
if {"Order Date","Sales"}.issubset(dff.columns):
    monthly = (
        dff.dropna(subset=["Order Date"])
           .set_index("Order Date")
           .resample("M")["Sales"].sum()
           .reset_index()
    )
    fig_ts = px.line(monthly, x="Order Date", y="Sales", markers=True)
    fig_ts.update_layout(yaxis_title="Sales", xaxis_title=None, margin=dict(t=10,b=10))
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("Kolom 'Order Date' dan/atau 'Sales' tidak tersedia.")

# ---------- TOP PRODUCTS TABLE ----------
st.subheader("🏆 Top 15 Products by Sales")
prod_cols = {"Product Name","Sales"}
# Kaggle file kadang pakai 'Product Name'
if "Product Name" not in dff.columns and "Product" in dff.columns:
    dff = dff.rename(columns={"Product": "Product Name"})
if prod_cols.issubset(dff.columns):
    top_products = (
        dff.groupby("Product Name", as_index=False)["Sales"]
        .sum().sort_values("Sales", ascending=False).head(15)
    )
    top_products["Sales"] = top_products["Sales"].round(0).apply(lambda v: f"${v:,.0f}")
    safe_dataframe(top_products, use_container_width=True, height=420)
else:
    st.info("Kolom 'Product Name' dan 'Sales' diperlukan untuk tabel ini.")

# ---------- CAPTION SUGGESTER ----------
with st.expander("📝 Caption IG/LinkedIn — siap tempel"):
    # hitung insight ringkas
    cat_name = "-"
    if "Category" in dff.columns and "Sales" in dff.columns and len(dff):
        cat_name = (
            dff.groupby("Category")["Sales"].sum().sort_values(ascending=False).index[0]
            if dff["Category"].notna().any() else "-"
        )
    region_name = "-"
    if "Region" in dff.columns and "Profit" in dff.columns and len(dff):
        region_name = (
            dff.groupby("Region")["Profit"].sum().sort_values(ascending=False).index[0]
            if dff["Region"].notna().any() else "-"
        )

    caption = f"""🚀 Day 1 of 100 Days of AI Projects

📊 Project: Analisis Dataset Penjualan (Kaggle – Superstore)
Hari pertama langsung pakai data nyata. Dengan Python (Pandas + Plotly), aku membangun mini dashboard untuk memantau Sales, Profit, dan tren bulanan.

✨ Insight singkat:
• Total Sales: {to_money(total_sales)}
• Total Profit: {to_money(total_profit)}
• Kategori terlaris: {cat_name}
• Region paling profit: {region_name}

Toolkit: Python, Pandas, Plotly, Streamlit
#100DaysOfAI #MachineLearning #DataScience #Python #Analytics #PersonalBranding
"""
    st.code(caption, language="markdown")

st.caption("© Day 1 — Build fast, share insight, ulang besok dengan visual yang lebih tajam 🚀")
