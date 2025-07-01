import streamlit as st
st.title("Investment Screening Assistant")
st.write("Upload company documents and let AI help evaluate whether the business meets your investment criteria!")
uploaded_file = st.file_uploader("Upload company profile, pitch deck, or business plan", type=["pdf", "pptx"])

# Text Extraction
def extract_text(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == "pdf":
        import fitz
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif file_type == "pptx":
        from pptx import Presentation
        prs = Presentation(uploaded_file)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    else:
        return "Unsupported file type."

# AI Analysis
import google.generativeai as genai
genai.configure(api_key=st.secrets["API"])
def analyze_with_ai(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    def split_text(text, chunk_size=3000):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for i, chunk in enumerate(split_text(text)):
        prompt = f"""
        This is part {i + 1} of a company document.
        Try to be concise and accurate. Do not omit any information that may affect investment decisions.
        Summarize the key points of this section only:

        {chunk}
        """
        response = model.generate_content(prompt)
        summaries.append(response.text)
    final_input = "\n\n".join(summaries)

    intro_prompt = f"""
    You are an investment analyst. Read the company content below and provide:
    
    - Company Description
    - Founders / Team
    - Main product/service
    - Key Features
    - Historical Financial Performance
    - Market Overview
    - Future Plan
    - Key Risks and Miligants
    
    Content:
    {final_input}
    """
    intro_response = model.generate_content(intro_prompt).text
    
    SEIS_promt = f"""
    Please assess whether the following company meets the eligibility criteria for the UK Enterprise Investment Scheme (EIS).
    Evaluate each criterion separately and respond with:
    ✅ Yes / ❌ No / ⚠️ Uncertain
    
    Criterion:
    UK-registered company
    Age ≤ 2 years
    Assets ≤ £350, 000
    Employees ≤ 25
    Risk-to-capital condition
    Business operates in a qualifying trade (not financial, real estate, energy generation, etc.)
    Funds will be used for eligible business purposes (not repaying debt or buying companies)
        
    Return as markdown table, with columns Criterion, Status, Explanation.
    Conclude whether the company is likely to qualify for EIS.
    
    Content:
    {final_input}
    """
    SEIS_response = model.generate_content(SEIS_promt).text

    EIS_promt = f"""
    Please assess whether the following company meets the eligibility criteria for the UK Enterprise Investment Scheme (EIS).
    
    Evaluate each criterion separately and respond with:
    ✅ Yes / ❌ No / ⚠️ Uncertain
    
    Criterion:
    UK-registered company
    Unlisted (or listed on AIM)
    Age ≤ 7 years
    Assets ≤ £15 million
    Employees ≤ 250
    Risk-to-capital condition
    Business operates in a qualifying trade (not financial, real estate, energy generation, etc.)
    Funds will be used for eligible business purposes (not repaying debt or buying companies)
    
    Return as markdown table, with columns Criterion, Status, Explanation.
    Conclude whether the company is likely to qualify for EIS.
    
    Content:
    {final_input}
    """
    EIS_response = model.generate_content(EIS_promt).text

    score_prompt = f"""
    Evaluate the company based on these criteria:
    1. Product  
    2. Team  
    3. Financial Assessment 
    4. Market Attractiveness  
    5. Exit Potential

    For each:
    - Score out of 10
    - 1-sentence explanation

    Return as markdown table. Show the final score out of 50. Conclude with investment recommendation.

    Content:
    {final_input}
    """
    score_response = model.generate_content(score_prompt).text

    return intro_response, SEIS_response, EIS_response, score_response

# Output
if uploaded_file:
    content = extract_text(uploaded_file)
    if content:
        st.text_area("Document Preview", content[:2000])
        if st.button("Analyze with AI"):
            with st.spinner("Analyzing..."):
                intro_response, SEIS_response, EIS_response, score_response = analyze_with_ai(content)
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

if "records" not in st.session_state:
    st.session_state["records"] = []
if st.button("Save Result"):
    st.session_state["records"].append({
        "filename": uploaded_file.name,
        "overview": intro_response,
        "score": score_response,
    })
    st.success("Result saved!")
st.subheader("Saved Analyses")
for idx, record in enumerate(st.session_state["records"]):
    with st.expander(f"{idx+1}. {record['filename']}"):
        st.markdown("**Company Overview:**")
        st.write(record["overview"])
        st.markdown("**Score Table:**")
        st.markdown(record["score"])






