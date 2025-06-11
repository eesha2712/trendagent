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
for file in uploaded_files or []:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        df.columns = [str(c).strip() for c in df.columns]
        datasets[file.name] = df
    else:
        xls = pd.read_excel(file, sheet_name=None)
        for sheet, df in xls.items():
            df.columns = [str(c).strip() for c in df.columns]
            datasets[f"{file.name} [{sheet}]"] = df

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
        st.markdown("#### 1. Select datasets/columns for analysis")
        n_select = st.number_input("How many datasets to analyze?", min_value=1, max_value=len(dataset_names), value=2)
        ds_choices = [st.selectbox(f"Dataset #{i+1}", dataset_names, key=f"cds_{i}") for i in range(n_select)]
        var_choices = [st.multiselect(f"Columns from {ds_choices[i]}", list(datasets[ds_choices[i]].columns), key=f"cvars_{i}") for i in range(n_select)]

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

            # UI for dashboard chart types
            all_group_cols = list(data_for_analysis.columns)
            group_by = st.selectbox("Group/segment by (for trends/pie):", ["<None>"] + all_group_cols, key="cross_groupby")
            chart_type = st.selectbox(
                "Choose dashboard chart type",
                [
                    "Grouped Line Chart", "Stacked Bar Chart", "Pie Chart",
                    "Regression Plot", "Scatter Plot", "Correlation Heatmap"
                ],
                key="cross_chart_type"
            )
            y_vars = st.multiselect("Y-axis variable(s):", [c for c in data_for_analysis.columns if c != group_by], default=[c for c in data_for_analysis.columns if c != group_by][:1], key="cross_yvars")

            run_chart = st.button("Run Visualization", key="run_chart_btn")

            if run_chart:
                if chart_type == "Grouped Line Chart" and group_by != "<None>" and y_vars:
                    for idx, y in enumerate(y_vars):
                        fig = px.line(data_for_analysis, x=group_by, y=y, color=group_by)
                        st.plotly_chart(fig, use_container_width=True, key=f"grouped_line_{y}_{idx}")
                elif chart_type == "Stacked Bar Chart" and group_by != "<None>" and y_vars:
                    melted = data_for_analysis.melt(id_vars=group_by, value_vars=y_vars, var_name="Variable", value_name="Value")
                    fig = px.bar(melted, x=group_by, y="Value", color="Variable", barmode="stack")
                    st.plotly_chart(fig, use_container_width=True, key="stacked_bar")
                elif chart_type == "Pie Chart" and group_by != "<None>":
                    for idx, y in enumerate(y_vars):
                        pie_df = data_for_analysis.groupby(group_by)[y].sum().reset_index()
                        fig = px.pie(pie_df, names=group_by, values=y, title=f"Pie: {y} by {group_by}")
                        st.plotly_chart(fig, use_container_width=True, key=f"pie_{y}_{idx}")
                elif chart_type == "Regression Plot" and len(y_vars) == 2:
                    x, y = data_for_analysis[y_vars[0]].dropna(), data_for_analysis[y_vars[1]].dropna()
                    mask = x.notnull() & y.notnull()
                    x, y = x[mask], y[mask]
                    if len(x) > 1:
                        model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
                        pred = model.predict(x.values.reshape(-1, 1))
                        fig = px.scatter(x=x, y=y, trendline="ols", title=f"Regression: {y_vars[0]} vs {y_vars[1]}")
                        st.plotly_chart(fig, use_container_width=True, key="regression")
                        st.success(f"y = {model.coef_[0]:.3f}x + {model.intercept_:.3f} (R²={model.score(x.values.reshape(-1, 1), y.values):.3f})")
                    else:
                        st.warning("Not enough data for regression.")
                elif chart_type == "Scatter Plot" and len(y_vars) == 2:
                    fig = px.scatter(data_for_analysis, x=y_vars[0], y=y_vars[1])
                    st.plotly_chart(fig, use_container_width=True, key="scatter")
                elif chart_type == "Correlation Heatmap" and len(y_vars) >= 2:
                    fig = px.imshow(data_for_analysis[y_vars].corr())
                    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")
                else:
                    st.warning("Please select the correct variables and grouping for your chosen chart.")

            # --- Interesting Statistical Tests (with CSV Download) ---
st.markdown("### Select Statistical Tests to Run")
test_options = [
    "Summary Stats",
    "Correlation Matrix",
    "Regression (2 numeric variables)",
    "T-Test (group vs numeric)",
    "ANOVA (group vs numeric)",
    "Chi-square (two categoricals)"
]
selected_tests = st.multiselect("Pick tests", test_options, default=["Summary Stats"])

test_results = []
test_csvs = {}

if st.button("Run Selected Tests", key="run_selected_tests"):
    # Summary
    if "Summary Stats" in selected_tests:
        desc = data_for_analysis.describe(include="all").T
        st.write(desc)
        test_results.append(("Summary Stats", desc.to_string()))
        test_csvs["Summary_Stats.csv"] = desc

    # Correlation
    if "Correlation Matrix" in selected_tests:
        corr = data_for_analysis.corr()
        st.dataframe(corr)
        test_results.append(("Correlation Matrix", corr.to_string()))
        test_csvs["Correlation_Matrix.csv"] = corr

    # Regression
    if "Regression (2 numeric variables)" in selected_tests and len(y_vars) == 2:
        x, y = data_for_analysis[y_vars[0]].dropna(), data_for_analysis[y_vars[1]].dropna()
        mask = x.notnull() & y.notnull()
        x, y = x[mask], y[mask]
        if len(x) > 1:
            model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
            out = f"y = {model.coef_[0]:.3f}x + {model.intercept_:.3f} (R²={model.score(x.values.reshape(-1, 1), y.values):.3f})"
            st.success(out)
            test_results.append(("Regression", out))
            reg_df = pd.DataFrame({"x": x, "y": y, "y_pred": model.predict(x.values.reshape(-1, 1))})
            test_csvs["Regression_Data.csv"] = reg_df
        else:
            st.warning("Need more data for regression.")

    # T-Test
    if "T-Test (group vs numeric)" in selected_tests:
        group_col = st.selectbox("T-test: Group column (categorical, 2 levels)", all_group_cols, key="ttest_group_col2")
        num_col = st.selectbox("T-test: Numeric column", [c for c in all_group_cols if pd.api.types.is_numeric_dtype(data_for_analysis[c])], key="ttest_num_col2")
        group_vals = data_for_analysis[group_col].dropna().unique()
        if len(group_vals) == 2:
            grp1 = data_for_analysis[data_for_analysis[group_col]==group_vals[0]][num_col].dropna()
            grp2 = data_for_analysis[data_for_analysis[group_col]==group_vals[1]][num_col].dropna()
            t_stat, p_val = stats.ttest_ind(grp1, grp2, equal_var=False)
            out = f"T-test: t={t_stat:.3f}, p={p_val:.4f} ({group_vals[0]} vs {group_vals[1]})"
            st.info(out)
            test_results.append(("T-test", out))
            ttest_df = pd.DataFrame({f"{group_col}": np.concatenate([np.repeat(group_vals[0], len(grp1)), np.repeat(group_vals[1], len(grp2))]),
                                    num_col: np.concatenate([grp1, grp2])})
            test_csvs["Ttest_Data.csv"] = ttest_df
        else:
            st.warning("Selected group column must have exactly 2 unique values for t-test.")

    # ANOVA
    if "ANOVA (group vs numeric)" in selected_tests:
        group_col = st.selectbox("ANOVA: Group column (categorical, >2 levels)", all_group_cols, key="anova_group_col2")
        num_col = st.selectbox("ANOVA: Numeric column", [c for c in all_group_cols if pd.api.types.is_numeric_dtype(data_for_analysis[c])], key="anova_num_col2")
        groups = data_for_analysis[group_col].dropna().unique()
        if len(groups) > 2:
            arrays = [data_for_analysis[data_for_analysis[group_col]==g][num_col].dropna() for g in groups]
            f_stat, p_val = stats.f_oneway(*arrays)
            out = f"ANOVA: F={f_stat:.3f}, p={p_val:.4f}"
            st.info(out)
            test_results.append(("ANOVA", out))
            anova_df = data_for_analysis[[group_col, num_col]].dropna()
            test_csvs["ANOVA_Data.csv"] = anova_df
        else:
            st.warning("Selected group column must have >2 unique values for ANOVA.")

    # Chi-square
    if "Chi-square (two categoricals)" in selected_tests:
        cat1 = st.selectbox("Chi-square: Categorical column 1", all_group_cols, key="chi_col1b")
        cat2 = st.selectbox("Chi-square: Categorical column 2", [c for c in all_group_cols if c != cat1], key="chi_col2b")
        tbl = pd.crosstab(data_for_analysis[cat1], data_for_analysis[cat2])
        chi2, p, dof, expected = stats.chi2_contingency(tbl)
        out = f"Chi-square: χ²={chi2:.2f}, p={p:.4f}, dof={dof}"
        st.info(out)
        test_results.append(("Chi-square", out))
        test_csvs["ChiSq_Data.csv"] = tbl

    # Download results table
    if test_results:
        results_df = pd.DataFrame(test_results, columns=["Test", "Result"])
        st.markdown("#### Download All Test Results")
        st.download_button("Download All Results (CSV)", results_df.to_csv(index=False).encode(), "test_results.csv")

        # Download individual CSVs for each test’s raw data
        for fname, df in test_csvs.items():
            st.download_button(f"Download {fname}", df.to_csv(index=True).encode(), file_name=fname)

    # Download all analyzed data
    st.markdown("#### Download Analyzed Data")
    st.download_button("Download Data (CSV)", data_for_analysis.to_csv(index=False).encode(), "analyzed_data.csv")

    # --- Cross-Analysis Gemini Q&A ---
st.markdown("#### Gemini AI: Insight from Cross-Dataset Analysis")
cross_q = st.text_input("Ask Gemini (Cross-Analysis)", key="cross_gem")
if st.button("Ask Gemini about Cross-Analysis", key="cross_gem_btn"):
    if data_for_analysis is not None and not data_for_analysis.empty:
        context = f"Columns: {', '.join(data_for_analysis.columns)}\nFirst few rows:\n{data_for_analysis.head(3).to_string(index=False)}"
        prompt = (
            f"You are an AI data analyst. Below is a combined dataset for cross-analysis:\n{context}\n\n"
            f"User question: {cross_q}\n\n"
            "Suggest interesting trends to investigate, tests to run (t-test, regression, chi-square, etc.), or help interpret dashboard graphs. "
            "Highlight possible group differences or relationships."
        )
        with st.spinner("Gemini is answering..."):
            st.info(gemini_chat(prompt))
    else:
        st.warning("No data available for cross-analysis.")

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
        aiq = st.text_input("Ask Gemini (eda tab)", key="eda_gem")
        if st.button("Ask on EDA", key="eda_gem_btn"):
            st.info(gemini_chat(f"EDA on {dsn} for columns {cols}. {aiq}"))

        st.markdown("**Coming soon:** Instant anomaly detection, pattern mining, feature engineering, explainable outlier detection.")
        st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=600&q=80", width=240, caption="EDA AI coming soon!")

    else:
        st.info("Upload data to begin EDA.")

    # --- EDA Gemini Q&A ---
st.markdown("#### Gemini AI: Exploratory Data Analysis Ideas")
eda_q = st.text_input("Ask Gemini (EDA)", key="eda_gem")
if st.button("Ask Gemini about EDA", key="eda_gem_btn"):
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
