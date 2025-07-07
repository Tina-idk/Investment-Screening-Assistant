import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np
import uuid

st.title("Investment Screening Assistant")
st.write("Upload company documents and let AI help evaluate whether the business meets your investment criteria!")
uploaded_file = st.file_uploader("Upload company profile, pitch deck, or business plan", type=["pdf"])

# Text Extraction        
def generate_multi_comparison_conclusion(records):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    companies_text = ""
    for i, record in enumerate(records, 1):
        companies_text += f"\nCompany {i} ({record['filename']}):\n{record['score']}\n"

    prompt = f"""
    Compare the following companies based on their investment scores.
    Each has been evaluated based on: Product, Team, Financial Assessment, Market Attractiveness, Exit Potential.

    {companies_text}

    Write a comparison summary (5–8 sentences) explaining:
    - Which companies perform better overall and why
    - Their respective strengths and weaknesses
    - If any company stands out as the best investment target
    - Any ties or close comparisons worth mentioning

    Be concise, neutral, and insightful.
    """
    response = model.generate_content(prompt, generation_config={"temperature": 0})
    return response.text
    
def parse_score_table_to_df(markdown_table_str):
    lines = markdown_table_str.strip().splitlines()
    table_started = False
    table_lines = []
    for line in lines:
        if line.strip().startswith("|"):
            table_started = True
            table_lines.append(line)
        elif table_started:
            break  
    table_lines = [line for line in table_lines if "---" not in line]
    if len(table_lines) < 2:
        return pd.DataFrame()
    
    rows = []
    for line in table_lines[1:]:  
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        rows.append(parts[:3])
    df = pd.DataFrame(rows, columns=["Criteria", "Score", "Explanation"])
    return df

def extract_scores_only(markdown_table_str):
    df = parse_score_table_to_df(markdown_table_str)
    if df.empty or "Score" not in df.columns or "Criteria" not in df.columns:
        return {}
    df["Score"] = pd.to_numeric(df["Score"].astype(str).str.strip(), errors='coerce')
    df["Score"] = df["Score"].fillna(0)
    df = df[df["Criteria"].isin([
        "Annual Revenue", "Growth", "Founders", "Market & Products",
        "Valuation Discipline", "Cap Table"
    ])]
    return df.set_index("Criteria")["Score"].to_dict()

def plot_radar_chart(scores_dict):
    fig = go.Figure()
    for name, scores in scores_dict.items():
        filtered_scores = {k: v for k, v in scores.items() if k != "Total Score"}
        criteria = list(filtered_scores.keys())
        values = list(filtered_scores.values())
        values += values[:1]  # 形成闭环
        criteria += criteria[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=criteria,
            fill='toself',
            name=name
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True
    )
    return fig

# AI Analysis
import google.generativeai as genai
genai.configure(api_key=st.secrets["API"])
def analyze_with_ai(file):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    def prompt_and_ask(prompt):
        response = model.generate_content([prompt, file], generation_config={"temperature": 0})
        return response.text

    # 1. Overview Summary
    overview_prompt = """
    You are an investment analyst. Read the company content below and provide:

    - Company Description
    - Founders / Team
    - Main product/service
    - Key Features
    - Historical Financial Performance
    - Market Overview
    - Future Plan
    - Key Risks and Mitigants
    """
    intro_response = prompt_and_ask(overview_prompt)

    # 2. SEIS
    seis_prompt = """
    Please assess whether the following company meets the eligibility criteria for the UK Seed Enterprise Investment Scheme (SEIS).
    Evaluate each criterion separately and respond with ✅ Yes / ❌ No / ⚠️ Uncertain.

    Criteria:
    - UK-registered company
    - Age ≤ 2 years
    - Assets ≤ £350,000
    - Employees ≤ 25
    - Risk-to-capital condition
    - Business operates in a qualifying trade (not financial, real estate, energy generation, etc.)
    - Funds will be used for eligible business purposes (not repaying debt or buying companies)

    Return a markdown table with Criterion, Status, and Explanation. Conclude whether the company is likely to qualify for SEIS.
    """
    SEIS_response = prompt_and_ask(seis_prompt)

    # 3. EIS
    eis_prompt = """
    Please assess whether the following company meets the eligibility criteria for the UK Enterprise Investment Scheme (EIS).
    Evaluate each criterion separately and respond with ✅ Yes / ❌ No / ⚠️ Uncertain.

    Criteria:
    - UK-registered company
    - Unlisted (or listed on AIM)
    - Age ≤ 7 years
    - Assets ≤ £15 million
    - Employees ≤ 250
    - Risk-to-capital condition
    - Business operates in a qualifying trade (not financial, real estate, energy generation, etc.)
    - Funds will be used for eligible business purposes (not repaying debt or buying companies)

    Return a markdown table with Criterion, Status, and Explanation. Conclude whether the company is likely to qualify for EIS.
    """
    EIS_response = prompt_and_ask(eis_prompt)

    # 4. Investment Criteria Evaluation
    score_prompt = """
    Evaluate the company based on these criteria and corresponding standards of evaluation after the colon:
    Remember if you think there exist vital drawbacks, please deduct scores and give the reasons in the explanation
    
    Annual Revenue: 
        Revenue below £1M and Portion of revenue is SaaS / Recurring revenue less than 50% → Score 0
        Revenue below £1M or Portion of revenue is SaaS / Recurring revenue less than 50% → Score 1~2
        Revenue below £1M but a clear path to £1m within 6 months and Portion of revenue is SaaS / Recurring revenue (50%) → Score 3~4
        £1M+ revenue and Portion of revenue is SaaS / Recurring revenue (50%) → Score 4~5
                    
    Growth: 
        Early-stage companies(whose revenue is sub £3m):
            Not show MoM growth → Score 0
            Show MoM growth → Score 1~3
            Show strong MoM growth or 50% YoY growth → Score 4~5
        
        Later-stage companies(whose revenue is above £3m)
            Not demonstrate solid YoY growth → Score 0
            Demonstrate solid YoY growth:
                Agency models → Score 0
                Business model benchmarks: (If the score reaches benchmark → Score 4-5, or → Score 1~3)
                    SaaS: 100% YoY growth
                    Marketplaces: 60–100% YoY growth
                    Transactional: ~20% YoY growth
            Exception: operate in a market with smaller growth (FMCG) – Company is doing 50% on top of the market average → Score 4~5
            
    Founders: 
        Founders do not have deep domain expertise and a strong marketing-led strategy → Score 0
        Founders have deep domain expertise or a strong marketing-led strategy → Score 1~3
        Founders have deep domain expertise and a strong marketing-led strategy → Score 4~5
        
    Market & Products:
        Market is not growing or shrinking; product lacks innovation; market size is insufficient to support target exit revenue and 3x return → Score 0
        Market shows some growth; product has some innovation; total addressable market (TAM) is reasonably large but expected market share is low (<1%); 3x return logic is unclear or weak → Score 1~3
        Clearly growing market; product is innovative and competitive; large market size (supporting £20m+ revenue); expected to capture 1–3% market share in 3–5 years; clear and reasonable data supporting 3x return → 4~5

    Valuation Discipline:
        Valuation is not based on actual trailing 12-month revenue multiples or lacks justification → Score 0
        Valuation mostly based on trailing 12-month revenues, but justification in due diligence (DD) report is weak or incomplete → Score 1~3
        Valuation firmly grounded in multiples of actual trailing 12-month revenues; justifications are clear, well-documented, and withstand rigorous scrutiny in DD → Score 4~5
        
    Cap Table:
        Founders does not retain at least 40% ownership pre-Series A (including option pool) → Score 0
        Founders retain at least 40% ownership pre-Series A (including option pool) → Score 4~5
        Exception: The Options pool of X is being created to bolster founder shareholder → Score 4~5
        
    For each:
    - Score out of 5
    - 1-sentence explanation (with numeric data)

    Return a markdown table. Calculate the final score out of 30. Conclude with investment recommendation.
    """
    score_runs = [prompt_and_ask(score_prompt) for _ in range(5)]
    score_response = score_runs[0]

    # 5. 评分平均处理
    from collections import defaultdict
    import numpy as np
    score_data = defaultdict(list)
    for output in score_runs:
        parsed = extract_scores_only(output)
        for k, v in parsed.items():
            score_data[k].append(v)
    averaged_scores = {k: round(np.mean(v)) for k, v in score_data.items()}
    averaged_scores["Total Score"] = round(sum([v for k, v in averaged_scores.items() if k != "Total Score"]), 2)

    return intro_response, SEIS_response, EIS_response, score_response, averaged_scores

# Output
# Only show "Save Result" if AI analysis has been run
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if "records" not in st.session_state:
    st.session_state["records"] = []

if uploaded_file and uploaded_file.name.lower().endswith(".pdf"):
    try:
        uploaded = genai.upload_file(
            uploaded_file.read(),  # 转成 bytes
            mime_type="application/pdf",
            file_name=f"{uuid.uuid4()}_{uploaded_file.name}"  # 避免文件名重复
        )
        st.success("✅ File uploaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to upload file to Gemini API: {e}")
    
    if st.button("Analyze with AI"):
        with st.spinner("Analyzing..."):
            intro_response, SEIS_response, EIS_response, score_response, averaged_scores = analyze_with_ai(uploaded)
            st.session_state["intro_response"] = intro_response
            st.session_state["score_response"] = score_response
            st.session_state["averaged_scores"] = averaged_scores
            st.session_state["analysis_done"] = True
        st.subheader("Company Overview")
        st.write(intro_response)
        st.subheader("SEIS Judgement")
        st.markdown(SEIS_response)
        st.subheader("EIS Judgement")
        st.markdown(EIS_response)
        st.subheader("Investment Criteria Evaluation")
        st.markdown(score_response)
    else:
        st.error("Unsupported file type or empty content.")

MAX_RECORDS = 10

# Only allow saving if analysis is completed
if st.session_state.get("analysis_done", False):
    if st.button("Save Result"):
        st.session_state["records"].append({
            "filename": uploaded_file.name,
            "overview": st.session_state["intro_response"],
            "score": st.session_state["score_response"],
            "avg_score": st.session_state["averaged_scores"]
        })
        if len(st.session_state["records"]) > MAX_RECORDS:
            st.session_state["records"] = st.session_state["records"][-MAX_RECORDS:]
        st.success("Result saved!")

if "records" in st.session_state and len(st.session_state["records"]) > 0:
    # Convert saved scores into a DataFrame for export
    if st.button("Export All Scores to CSV"):
        data = []
        for record in st.session_state["records"]:
            avg_score = record.get("avg_score", {})
            if avg_score:
                score_data = avg_score.copy()
            else:
                score_data = extract_scores_only(record["score"])
                score_data["Total Score"] = sum(score_data.values())
            score_data["Company"] = record["filename"]
            data.append(score_data)
            
        df_export = pd.DataFrame(data)
        df_export = df_export[["Company"] + [col for col in df_export.columns if col != "Company"]]
        df_export["Recommended"] = ""  
        st.dataframe(df_export)
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV for ML Training",
            csv,
            file_name="investment_scores.csv",
            mime="text/csv"
        )

    options = [f"{i+1}. {record['filename']}" for i, record in enumerate(st.session_state["records"])]
    default_selection = options[:2] if len(options) >= 2 else options

    selected = st.multiselect(
        "Select companies to compare (up to 10)",
        options,
        default=default_selection,
        max_selections=10
    )

    if len(selected) >= 2:
        indices = [int(s.split(".")[0]) - 1 for s in selected]
        records = [st.session_state["records"][i] for i in indices]
    
        score_dict = {}
        for record in records:
            avg_score = record.get("avg_score")
            if avg_score:
                score_dict[record["filename"]] = avg_score
            else:
                score_dict[record["filename"]] = extract_scores_only(record["score"])

        score_dict_radar = {
            name: {k: v for k, v in scores.items() if "total" not in k.lower()}
            for name, scores in score_dict.items()
        }

        criteria_order = [
            "Annual Revenue", "Growth", "Founders", "Market & Products",
            "Valuation Discipline", "Cap Table", "Total Score"
        ]
    
        df = pd.DataFrame(score_dict)
    
        df = df.reindex(criteria_order)
    
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
    
        numeric_df.loc["Total Score"] = numeric_df.drop("Total Score", errors='ignore').sum()
    
        st.markdown("### Score Table Comparison")
        st.dataframe(numeric_df.T, use_container_width=True)

        st.markdown("### Radar Chart: Criteria Overview")
        radar_fig = plot_radar_chart(score_dict_radar)
        st.plotly_chart(radar_fig, use_container_width=True)
    
        combo_key = frozenset([record["filename"] for record in records])
        if "multi_conclusions" not in st.session_state:
            st.session_state["multi_conclusions"] = {}
        if combo_key in st.session_state["multi_conclusions"]:
            summary = st.session_state["multi_conclusions"][combo_key]
        else:
            summary = generate_multi_comparison_conclusion(records)
            st.session_state["multi_conclusions"][combo_key] = summary
    
        st.subheader("AI Summary Conclusion")
        st.write(summary)
    
        st.markdown("### Score Breakdown per Company")
        for record in records:
            with st.expander(f"View detailed explanations for: {record['filename']}"):
                score_df = parse_score_table_to_df(record["score"])
                if not score_df.empty:
                    score_df = score_df[["Criteria", "Explanation"]].set_index("Criteria")
                    st.table(score_df)
                else:
                    st.info("No explanation found.")

    elif len(selected) == 1:
        idx = int(selected[0].split(".")[0]) - 1
        record = st.session_state["records"][idx]
        st.markdown(f"###  {record['filename']}")
        st.markdown(record["score"])
        
else:
    st.info("Please upload at least one company file.")


if st.button("Clear All Saved Analyses"):
    st.session_state["records"] = []
    st.success("All records have been cleared!")
