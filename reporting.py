import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO

# Import the shared API handler from the data_sourcing module
from data_sourcing import handle_api_request

def generate_report_api(cleaned_text, sentiment_results, market_change, api_key):
    """Generates a summary report using the xAI API."""
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    prompt = f"""
    **Objective:** Generate a comprehensive, professional financial report based on the provided earnings call transcript analysis.
    **Report Structure:**
    1.  **Executive Summary:** A brief, high-level overview of the earnings call's key themes, overall sentiment, and subsequent market reaction.
    2.  **Key Topics Discussed:** Identify and summarize the 3-5 most critical topics from the call (e.g., revenue growth, product performance, future guidance, challenges ). Use bullet points for clarity.
    3.  **Sentiment Analysis Deep Dive:**
        *   **Overall Tone:** Describe the general sentiment (positive, negative, neutral, mixed) of the call.
        *   **Methodology Explanation:** Briefly explain what FinBERT, VADER, and the Loughran-McDonald (LM) lexicons measure in a financial context.
        *   **Results Interpretation:** Analyze the provided sentiment scores:
            *   FinBERT Score: {sentiment_results['FinBERT']['score']:.3f}
            *   VADER Score: {sentiment_results['VADER']['score']:.3f}
            *   LM Score: {sentiment_results['LM']['score']:.3f}
            *   Discuss any convergence or divergence between the models. For instance, 'All three models indicated a positive tone, with FinBERT showing the strongest signal.'
    4.  **Market Reaction Analysis:**
        *   The stock's price changed by **{market_change:.2f}%** in the 7 days following the call.
        *   Interpret this movement in the context of the sentiment scores. Did the market react in line with the call's sentiment? Discuss potential reasons for any discrepancies (e.g., broader market trends, pre-announcement expectations).
    5.  **Conclusion:** A concluding paragraph summarizing the findings and the overall picture of the company's performance and outlook as presented in the call.
    **Source Transcript (first 1000 characters for context):**
    ---
    {cleaned_text[:1000]}...
    ---
    Please generate the full report based on this structure and data.
    """
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}]}

    data = handle_api_request(url, payload, headers)

    if data and data.get("choices"):
        return data["choices"][0]["message"]["content"]
    st.error("Failed to generate report using API.")
    return "Report generation failed."

def create_pdf_report(report_text, figs):
    """Builds a PDF report with text and figures."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Earnings Call Sentiment Analysis Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    report_lines = report_text.split('\n')
    for line in report_lines:
        line = line.strip()
        if line.startswith('**') and line.endswith('**'):
            story.append(Paragraph(line.replace('**', ''), styles['h2']))
            story.append(Spacer(1, 0.1 * inch))
        elif line:
            story.append(Paragraph(line, styles['BodyText']))
            story.append(Spacer(1, 0.05 * inch))

    if figs:
        story.append(PageBreak())
        story.append(Paragraph("Visualizations", styles['h1']))
        story.append(Spacer(1, 0.2 * inch))

        fig_order = ['word_cloud', 'score_comparison', 'market_performance']
        for fig_name in fig_order:
            if fig_name in figs:
                fig_obj = figs[fig_name]
                img_buffer = BytesIO()
                fig_obj.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)

                img = Image(img_buffer, width=6*inch, height=4*inch, kind='proportional')
                story.append(img)
                story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer
