import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.express as px
import requests
import json

# --- Gemini AI helper ---
def gemini_chat(prompt):
    API_KEY = "AIzaSyBPmuR4UwM4zML8UWsJaIivGICSP1s4O1Y"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        resp.raise_for_status()
        candidates = resp.json().get("candidates")
        if candidates and "content" in candidates[0]:
            return candidates[0]["content"]["parts"][0]["text"]
        elif candidates and "output" in candidates[0]:
            return candidates[0]["output"]
        else:
            return "No answer returned."
    except Exception as e:
        return f"Gemini error: {e}"

st.set_page_config(page_title="TrendAgent", layout="wide", page_icon="📊")

st.markdown(
    "<h1 style='text-align: center;'>TrendAgent</h1>"
    "<h4 style='text-align: center; color: #64748b;'>AI Agent for Instant EDA, Trend Flagging, and Group Insights</h4>",
    unsafe_allow_html=True,
)

# ---- Sidebar ----
with st.sidebar:
    st.header("Upload Datasets")
    uploaded_files = st.file_uploader("Upload CSV or Excel", accept_multiple_files=True, type=["csv", "xlsx"])
    st.markdown(
        "<small><b>Trendency AI Vision:</b><br>AI for multi-survey trend detection, scenario simulation, group insights, and instant report generation.</small>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.subheader("💬 Freeform Gemini Q&A")
    freeform_q = st.text_area("Ask Gemini anything about your data, analysis, or trends.", key="sidebar_gemini_input")
    if st.button("Ask Gemini", key="sidebar_gemini_btn"):
        if uploaded_files:
            context = "\n\n".join(
                [f"{name}:\n{pd.read_csv(f).head().to_string() if name.endswith('.csv') else pd.read_excel(f).head().to_string()}"
                for name, f in zip([file.name for file in uploaded_files], uploaded_files)]
            )
            full_prompt = f"{freeform_q}\n\nContext:\n{context}"
            with st.spinner("Gemini is answering..."):
                st.info(gemini_chat(full_prompt))
        else:
            st.warning("Upload at least one dataset for Gemini context.")

# ---- Load Data ----
datasets = {}
if uploaded_files:
    for file in uploaded_files:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                df.columns = [str(c).strip() for c in df.columns]
                datasets[file.name] = df
            else:
                xls = pd.read_excel(file, sheet_name=None)
                for sheet, df in xls.items():
                    df.columns = [str(c).strip() for c in df.columns]
                    datasets[f"{file.name} [{sheet}]"] = df
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")

tab_titles = [
    "📚 Data Dictionary",
    "📈 Cross-Dataset Trends & Groups",
    "🧭 EDA",
    "🧠 AI Features & Anomalies (Coming Soon)",
    "📝 Report Generation (Coming Soon)",
    "🌐 External Data Integration (Coming Soon)"
]
tabs = st.tabs(tab_titles)

# --- 1. Data Dictionary ---
with tabs[0]:
    st.subheader("Data Dictionary")
    with st.expander("ℹ️ What does this do?"):
        st.markdown("""
- Summarizes your uploaded datasets (columns, type, uniqueness, missing, and sample values) in a single table.
- Essential for seeing what’s available, prepping merges, and exploring before any analysis.
        """)
    if datasets:
        for fname, df in datasets.items():
            st.markdown(f"**{fname}**")
            summary = pd.DataFrame({
                "Column": df.columns,
                "Type": [str(df[c].dtype) for c in df.columns],
                "Unique": [df[c].nunique() for c in df.columns],
                "Missing": [df[c].isnull().sum() for c in df.columns],
                "Sample": [str(df[c].dropna().iloc[0]) if df[c].dropna().size else "" for c in df.columns]
            })
            st.dataframe(summary, use_container_width=True, key=f"dict_{fname}")
    else:
        st.info("Upload files to see a dictionary of all columns/types/uniques.")

    st.markdown("---")
    st.markdown("**Gemini AI Q&A:** Ask about your schema or next steps.")
    q = st.text_input("Ask Gemini (dictionary tab)", key="dict_gem")
    if st.button("Ask on Dictionary", key="dict_gem_btn"):
        if datasets:
            schema_str = "\n".join([f"{k}: {','.join(list(v.columns))}" for k,v in datasets.items()])
            prompt = f"User schema: {schema_str}\nUser question: {q}"
            with st.spinner("Gemini is answering..."):
                st.info(gemini_chat(prompt))
        else:
            st.warning("No data to analyze.")

    st.markdown("**Coming soon:** Data profiling, codebook auto-generation, interactive schema editing.")
    st.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", width=220, caption="Dictionary AI coming soon!")

# --- 2. Cross-Dataset Trends & Groups ---
with tabs[1]:
    st.subheader("Cross-Dataset Trend Analysis & Dashboard Graphs")
    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
- Select any number of datasets and any columns for analysis. No need to merge—just compare or trend any variables side by side!
- Easily plot trends by group—see time, demographic, or region lines/stacked bars, grouped lines, pie charts, regression, ANOVA, t-test, chi-square in a single click!
        """)
    if datasets:
        dataset_names = list(datasets.keys())
        max_n = len(dataset_names)
        if max_n == 1:
            n_select = 1
        else:
            n_select = st.number_input(
                "How many datasets to analyze?",
                min_value=1,
                max_value=max_n,
                value=min(2, max_n),
                step=1,
                key="n_select_input"
            )
        ds_choices = [st.selectbox(f"Dataset #{i+1}", dataset_names, key=f"cds_{i}") for i in range(int(n_select))]
        var_choices = [st.multiselect(f"Columns from {ds_choices[i]}", list(datasets[ds_choices[i]].columns), key=f"cvars_{i}") for i in range(int(n_select))]

        # Merge suggestion
        common_keys = set(datasets[ds_choices[0]].columns)
        for d in ds_choices[1:]:
            common_keys = common_keys & set(datasets[d].columns)
        suggest_merge = len(ds_choices) > 1 and bool(common_keys)
        do_merge = False
        join_key = None
        if suggest_merge:
            st.info(f"Suggested merge possible on columns: {', '.join(common_keys)}")
            do_merge = st.checkbox("Merge datasets on a common key for row-level analysis?", value=False)
            join_key = st.selectbox("Select key to merge on", list(common_keys), key="join_key") if do_merge else None

        # Prepare data for analysis
        if do_merge and join_key:
            dfs = [datasets[ds][[join_key] + vars] for ds, vars in zip(ds_choices, var_choices)]
            merged = dfs[0]
            for i, df in enumerate(dfs[1:]):
                merged = merged.merge(df, on=join_key, how="outer", suffixes=('', f'_other_{i}'))
            st.success(f"Datasets merged on key: {join_key}")
            data_for_analysis = merged
        else:
            cols = {}
            for ds, vars in zip(ds_choices, var_choices):
                for v in vars:
                    cols[f"{ds}::{v}"] = datasets[ds][v].reset_index(drop=True)
            data_for_analysis = pd.DataFrame(cols)

        # Ensure not empty
        if data_for_analysis is not None and not data_for_analysis.empty and data_for_analysis.shape[1] > 0:
            st.markdown("### Data Snapshot")
            st.dataframe(data_for_analysis.head(20), use_container_width=True, key="cross_data_snapshot")
            # ...keep your plotting, stats, and Gemini logic as before...
        else:
            data_for_analysis = None
    else:
        st.info("Upload at least one dataset to start analysis.")

    # (Rest of your charting/stats/Gemini code can remain unchanged!)

# --- 3. EDA ---
with tabs[2]:
    st.subheader("Exploratory Data Analysis (EDA)")
    with st.expander("ℹ️ How does EDA fit in?"):
        st.markdown("""
- Drill down into a single dataset, plot and summarize any variable(s), grouped or not.
- See outliers, distributions, and subgroup summaries to prep for deeper analysis.
        """)
    if datasets:
        dsn = st.selectbox("Select dataset", list(datasets.keys()), key="eda_ds")
        df = datasets[dsn]
        st.write(df.head())
        cols = st.multiselect("Columns to analyze", list(df.columns), default=list(df.columns)[:2], key="eda_cols")
        grp_col = st.selectbox("Group by (optional)", ["<None>"] + list(df.columns), key="eda_grp")
        plot_type = st.selectbox("Plot type", ["Histogram", "Boxplot", "Line", "Scatter", "Bar"], key="eda_plot_type")
        run_eda = st.button("Run EDA Visualization", key="run_eda_btn")
        if cols and run_eda:
            for i, c in enumerate(cols):
                if grp_col != "<None>":
                    if plot_type == "Histogram":
                        fig = px.histogram(df, x=c, color=grp_col, barmode="overlay")
                    elif plot_type == "Boxplot":
                        fig = px.box(df, x=grp_col, y=c)
                    elif plot_type == "Line":
                        fig = px.line(df.sort_values(grp_col), x=grp_col, y=c)
                    elif plot_type == "Scatter" and len(cols) == 2:
                        fig = px.scatter(df, x=cols[0], y=cols[1], color=grp_col)
                    elif plot_type == "Bar":
                        fig = px.bar(df, x=grp_col, y=c)
                    else:
                        continue
                else:
                    if plot_type == "Histogram":
                        fig = px.histogram(df, x=c)
                    elif plot_type == "Boxplot":
                        fig = px.box(df, y=c)
                    elif plot_type == "Line":
                        fig = px.line(df.sort_values(c), y=c)
                    elif plot_type == "Scatter" and len(cols) == 2:
                        fig = px.scatter(df, x=cols[0], y=cols[1])
                    elif plot_type == "Bar":
                        fig = px.bar(df, x=c)
                    else:
                        continue
                st.plotly_chart(fig, use_container_width=True, key=f"eda_{plot_type}_{c}_{i}")
        st.markdown("#### Gemini AI: Ask for EDA suggestions or interpretations.")
        aiq = st.text_input("Ask Gemini (eda tab)", key="eda_gem_tab")
        if st.button("Ask on EDA", key="eda_gem_btn_tab"):
            st.info(gemini_chat(f"EDA on {dsn} for columns {cols}. {aiq}"))

        st.markdown("**Coming soon:** Instant anomaly detection, pattern mining, feature engineering, explainable outlier detection.")
        st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=600&q=80", width=240, caption="EDA AI coming soon!")

    else:
        st.info("Upload data to begin EDA.")

    # --- EDA Gemini Q&A ---
st.markdown("#### Gemini AI: Exploratory Data Analysis Ideas")
eda_q = st.text_input("Ask Gemini (EDA)", key="eda_gem_bottom")
if st.button("Ask Gemini about EDA", key="eda_gem_btn_bottom"):
    if 'df' in locals() and not df.empty:
        context = f"Columns: {', '.join(df.columns)}\nFirst few rows:\n{df.head(3).to_string(index=False)}"
        prompt = (
            f"You are an AI data analyst. Here is a sample from the current dataset:\n{context}\n\n"
            f"User question: {eda_q}\n\n"
            "Recommend useful visualizations, potential outlier checks, or describe notable patterns by group. "
            "If asked, explain why a particular test or plot is appropriate."
        )
        with st.spinner("Gemini is answering..."):
            st.info(gemini_chat(prompt))
    else:
        st.warning("Upload a dataset and select columns for EDA.")

# --- 4. AI Features/Anomalies (Coming Soon) ---
with tabs[3]:
    st.subheader("AI Features: Anomaly Detection, Pattern Mining, Explainable Outlier Detection")
    st.markdown("🚧 **Coming Soon:** This area will let you run:")
    st.markdown("- Instant anomaly detection")
    st.markdown("- Pattern mining for hidden trends")
    st.markdown("- Automated feature engineering")
    st.markdown("- Explainable outlier detection")
    st.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", width=220, caption="Advanced AI Insights coming soon!")
    st.button("Run Anomaly Detection (Coming Soon)", disabled=True)
    st.button("Run Pattern Mining (Coming Soon)", disabled=True)
    st.button("Run Feature Engineering (Coming Soon)", disabled=True)
    st.button("Run Explainable Outlier Detection (Coming Soon)", disabled=True)

# --- 5. Report Generation (Coming Soon) ---
with tabs[4]:
    st.subheader("Automated Report Generation")
    st.markdown("*(Coming soon)* — Auto-produce reports (PDF/Word/PowerPoint) with charts, insights, and plain-English takeaways.")
    st.button("Generate Full Report (Coming Soon)", disabled=True)
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMWtrNWtmZDBmZjJmZTN0NmQyM2QwZWFsZzE0N3d4dXZ2NW11b2d3dCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/WoD6JZnwap6s8/giphy.gif", width=250, caption="Automated Reporting AI coming soon!")

# --- 6. External Data Integration (Coming Soon) ---
with tabs[5]:
    st.subheader("External Data Integration")
    st.markdown("*(Coming soon)* — Enrich with real-time news, economic indicators, or 3rd-party APIs for deeper context and trend discovery.")
    st.button("Integrate External Data (Coming Soon)", disabled=True)
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZm1ha3AzcWYyYjJhbnJhbGxubjlwM2Zoa2dsamdsbWJoemUybDhlZiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/ZCNGjP1BUFR2O/giphy.gif", width=220, caption="External Data AI coming soon!")

st.markdown("<hr><center><small style='color:#64748b'>POC © 2024 Eesha Iyer | Trendency</small></center>", unsafe_allow_html=True)
