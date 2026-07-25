"""
Reinstatly - Amazon POA Builder
--------------------------------
MODIFICATION NOTE:
Swapped the local Ollama backend for Groq's hosted API (langchain-groq using llama-3.3-70b-versatile).
This enables deployment on Streamlit Community Cloud.
- Reads API key from st.secrets["GROQ_API_KEY"].
- Handles Groq rate limit/API errors gracefully.
- All guardrails, anti-fabrication checks, prompts, and UI flows remain untouched.
"""

import streamlit as st
import json
import hashlib
import re
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# CONSTANTEN & INITIALISATIE
# ==========================================
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2
LOG_FILE = "sessions_log.json"

DISCLAIMER_TEXT = (
    "⚠️ **Disclaimer:** This tool drafts a starting point based only on what you provide. "
    "It is not legal advice, does not guarantee reinstatement, and does not replace "
    "professional review for complex cases (IP disputes, related-account issues, or anything involving legal risk)."
)

# Initialize Groq LLM
@st.cache_resource
def get_llm():
    if "GROQ_API_KEY" not in st.secrets:
        return None
    
    return ChatGroq(
        model=MODEL_NAME, 
        temperature=TEMPERATURE,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

# ==========================================
# HULPFUNCTIES & GUARDRAILS
# ==========================================

def is_english(text: str) -> bool:
    """Basis taaldetectie op basis van ASCII-verhouding."""
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        non_ascii = len([char for char in text if ord(char) > 128])
        return (non_ascii / len(text)) < 0.15

def log_session(notice_text: str, category: str):
    """Slaat geanonimiseerde sessielogs op."""
    notice_hash = hashlib.sha256(notice_text.encode('utf-8')).hexdigest()
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "notice_hash": notice_hash,
        "assigned_category": category
    }
    
    try:
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
            
        logs.append(log_entry)
        
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=4)
    except Exception as e:
        st.error(f"Fout bij het opslaan van log: {e}")

def sanitize_and_check_draft(draft_text: str, combined_user_inputs: str) -> tuple[str, bool]:
    """
    STAP A, B, C & D:
    Detecteert gefabriceerde feiten (bronnen, aantallen, datums) en vervangt ze
    direct door bracketed placeholders in de gegenereerde draft.
    """
    user_input_lower = combined_user_inputs.lower()
    sanitized_draft = draft_text
    replacements_made = False

    # --- 1. GETALLEN & UITGESCHREVEN GETALLEN ---
    spelled_numbers = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
        "eighteen", "nineteen", "twenty"
    ]
    
    quantity_patterns = [
        r'\b(?:\d+|' + '|'.join(spelled_numbers) + r')\s+(?:remaining\s+)?units\b',
        r'\b(?:\d+|' + '|'.join(spelled_numbers) + r')\s+items\b',
        r'\b(?:\d+|' + '|'.join(spelled_numbers) + r')\s+products\b'
    ]
    
    for pattern in quantity_patterns:
        matches = re.findall(pattern, sanitized_draft, flags=re.IGNORECASE)
        for match in matches:
            if match.lower() not in user_input_lower:
                sanitized_draft = re.sub(re.escape(match), "[insert exact quantity of units]", sanitized_draft, flags=re.IGNORECASE)
                replacements_made = True

    # --- 2. BRONNEN / INKOOP NARRATIEVEN ---
    sourcing_keywords = [
        r'\b(?:an?\s+)?online\s+liquidation\s+(?:website|source|retailer)\b',
        r'\bliquidation\s+website\b',
        r'\bunauthorized\s+(?:distributor|wholesaler|supplier)\b',
        r'\bthird-party\s+liquidator\b',
        r'\bretail\s+arbitrage\s+source\b'
    ]
    
    for pattern in sourcing_keywords:
        matches = re.findall(pattern, sanitized_draft, flags=re.IGNORECASE)
        for match in matches:
            if match.lower() not in user_input_lower:
                sanitized_draft = re.sub(re.escape(match), "[insert your actual product source/supplier]", sanitized_draft, flags=re.IGNORECASE)
                replacements_made = True

    # --- 3. SPECIFIEKE MERKBEWERINGEN & BELEID ---
    brand_claims = [
        r'\bstrictly\s+enforces\s+its?\s+distribution\s+rights\b',
        r'\bdoes\s+not\s+allow\s+reselling\b'
    ]
    
    for pattern in brand_claims:
        matches = re.findall(pattern, sanitized_draft, flags=re.IGNORECASE)
        for match in matches:
            if match.lower() not in user_input_lower:
                sanitized_draft = re.sub(re.escape(match), "[insert brand's specific enforcement policy if known]", sanitized_draft, flags=re.IGNORECASE)
                replacements_made = True

    # --- 4. DATUMS ---
    date_pattern = r'\b(?:\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?)\b'
    dates_found = re.findall(date_pattern, sanitized_draft, flags=re.IGNORECASE)
    for date_str in dates_found:
        if date_str.lower() not in user_input_lower:
            sanitized_draft = re.sub(re.escape(date_str), "[insert date]", sanitized_draft, flags=re.IGNORECASE)
            replacements_made = True

    return sanitized_draft, replacements_made

# ==========================================
# LLM CALLS
# ==========================================

def run_diagnosis(notice_text: str, seller_type: str, user_cause: str, user_actions: str):
    """LLM Call #1: Diagnoseert de schorsing via Groq."""
    llm = get_llm()
    if not llm:
        return None
    
    system_prompt = (
        "You are an expert Amazon account reinstatement analyst.\n"
        "STRICT RULE: ONLY classify the root cause based on the literal text of the notice provided. "
        "Never assume facts, dates, product categories, or history not explicitly stated by the user.\n\n"
        "Allowed Categories:\n"
        "- Inauthenticity/IP complaint\n"
        "- Order Defect Rate / performance metrics\n"
        "- Related account\n"
        "- Restricted category/listing policy\n"
        "- Review manipulation\n"
        "- Other/Unclear\n\n"
        "If the notice text does not clearly indicate a category, output 'Unclear — recommend the seller re-read their notice for the specific policy section cited'.\n\n"
        "Output Format:\n"
        "Category: [Selected Category]\n"
        "Explanation: [2-3 sentence plain-English explanation quoting ONLY phrases from the notice text]."
    )
    
    user_message = f"Notice Text:\n{notice_text}\n\nSeller Type: {seller_type}\nUser Supposed Cause: {user_cause}\nActions Taken: {user_actions}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"input": user_message})
        return response.content
    except Exception as e:
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "429" in error_msg:
            st.error("⏳ This tool is getting more traffic than expected — please try again in a minute.")
        else:
            st.error("⚠️ An error occurred while contacting the AI service. Please try again later.")
        return None

def run_poa_generation(notice_text: str, seller_type: str, user_cause: str, user_actions: str, diagnosis: str):
    """LLM Call #2: Genereert het Plan of Action via Groq."""
    llm = get_llm()
    if not llm:
        return None
    
    system_prompt = (
        "You are a strict Amazon Appeal Drafting Assistant.\n\n"
        "ABSUTE REQUIREMENT - ZERO NARRATIVE FABRICATION:\n"
        "- You must NEVER invent or assume story details, sourcing methods (e.g., 'liquidation website', 'retail arbitrage'), "
        "supplier names, unit counts (e.g., '15 units'), dates, or brand enforcement habits unless explicitly provided in the input.\n"
        "- If a fact is missing, DO NOT make up a plausible narrative. Leave a bracketed placeholder instead, such as "
        "'[insert supplier name]', '[insert exact unit count]', or '[insert sourcing channel]'.\n\n"
        "STRUCTURE:\n"
        "Generate a structured Plan of Action (POA) with EXACTLY three sections:\n"
        "1. Greater Detail on the Root Cause of the Issue\n"
        "2. Immediate Corrective Actions Taken\n"
        "3. Preventive Measures Implemented\n\n"
        "TONE:\n"
        "Keep the tone strictly factual, professional, and objective. Avoid emotional, pleading, or apologetic phrases."
    )
    
    user_message = (
        f"Notice Text: {notice_text}\n"
        f"Seller Type: {seller_type}\n"
        f"Perceived Cause: {user_cause if user_cause else 'None provided'}\n"
        f"Actions Already Taken: {user_actions if user_actions else 'None provided'}\n"
        f"Confirmed Diagnosis: {diagnosis}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"input": user_message})
        return response.content
    except Exception as e:
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "429" in error_msg:
            st.error("⏳ This tool is getting more traffic than expected — please try again in a minute.")
        else:
            st.error("⚠️ An error occurred while contacting the AI service. Please try again later.")
        return None

# ==========================================
# STREAMLIT UI
# ==========================================

def main():
    st.set_page_config(page_title="Reinstatly - Amazon POA Builder", page_icon="📦", layout="centered")
    
    st.title("📦 Reinstatly")
    st.caption("Privacy-First Amazon Suspension Plan of Action (POA) Generator")
    
    # Check GROQ API Key Startup Guardrail
    if "GROQ_API_KEY" not in st.secrets or not st.secrets["GROQ_API_KEY"]:
        st.error("❌ **GROQ_API_KEY not configured.** Please add your Groq API key to Streamlit secrets.")
        st.stop()

    st.info(DISCLAIMER_TEXT)
    st.divider()

    if "step" not in st.session_state:
        st.session_state.step = 1
    if "diagnosis" not in st.session_state:
        st.session_state.diagnosis = None
    if "diagnosis_confirmed" not in st.session_state:
        st.session_state.diagnosis_confirmed = False
    if "poa_draft" not in st.session_state:
        st.session_state.poa_draft = None
    if "replacements_were_made" not in st.session_state:
        st.session_state.replacements_were_made = False

    # --------------------------------------------------
    # STEP 1: INTAKE
    # --------------------------------------------------
    st.header("Step 1: Intake")
    
    notice_text = st.text_area(
        "Paste your exact Amazon suspension/deactivation notice here *",
        height=200,
        placeholder="Dear Seller, Your account has been deactivated because..."
    )
    
    seller_type = st.selectbox(
        "What type of seller are you?",
        ["Private label", "Wholesale", "Retail arbitrage", "Dropshipping", "Other"]
    )
    
    user_cause = st.text_input("In your own words, what do you think caused this? (optional)")
    user_actions = st.text_area("What corrective actions, if any, have you already taken? (optional)")

    if st.button("Analyze Notice & Diagnose"):
        if not notice_text or len(notice_text.strip()) < 50:
            st.error("❌ Please paste the full suspension notice (at least 50 characters required).")
        elif not is_english(notice_text):
            st.warning("⚠️ Non-English text detected. Reinstatly currently only supports English suspension notices.")
        else:
            with st.spinner("Analyzing notice with AI..."):
                diag_result = run_diagnosis(notice_text, seller_type, user_cause, user_actions)
                if diag_result:
                    st.session_state.diagnosis = diag_result
                    st.session_state.step = 2
                    st.session_state.diagnosis_confirmed = False
                    
                    cat_match = re.search(r'Category:\s*(.*)', diag_result)
                    category_found = cat_match.group(1) if cat_match else "Unclear"
                    log_session(notice_text, category_found)
                    st.rerun()

    # --------------------------------------------------
    # STEP 2: ROOT CAUSE DIAGNOSIS
    # --------------------------------------------------
    if st.session_state.step >= 2 and st.session_state.diagnosis:
        st.divider()
        st.header("Step 2: Root Cause Diagnosis")
        
        st.subheader("AI Analysis Result:")
        st.write(st.session_state.diagnosis)
        
        st.write("---")
        st.write("**Does this diagnosis match your situation?**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, continue to POA Draft"):
                st.session_state.diagnosis_confirmed = True
                st.session_state.step = 3
                st.session_state.poa_draft = None  # Reset draft forces regeneration
                st.rerun()
                
        with col2:
            if st.button("❌ No, let me re-describe it"):
                st.session_state.diagnosis_confirmed = False
                st.session_state.step = 1
                st.info("Please update the details in Step 1 and run the analysis again.")

    # --------------------------------------------------
    # STEP 3: PLAN OF ACTION DRAFT
    # --------------------------------------------------
    if st.session_state.step == 3:
        st.divider()
        st.header("Step 3: Plan of Action Draft")
        
        if not st.session_state.diagnosis_confirmed:
            st.warning("Please confirm the diagnosis in Step 2 before generating the draft.")
        else:
            if not st.session_state.poa_draft:
                with st.spinner("Generating structured Plan of Action..."):
                    raw_draft = run_poa_generation(
                        notice_text, seller_type, user_cause, user_actions, st.session_state.diagnosis
                    )
                    
                    if raw_draft:
                        combined_inputs = f"{notice_text} {user_cause} {user_actions}"
                        clean_draft, replacements_made = sanitize_and_check_draft(raw_draft, combined_inputs)
                        
                        st.session_state.poa_draft = clean_draft
                        st.session_state.replacements_were_made = replacements_made

            if st.session_state.poa_draft:
                if st.session_state.replacements_were_made:
                    st.warning(
                        "⚠️ **Some details in your draft are placeholders because you didn't provide them** — "
                        "fill these in with your real information before submitting. Never let the AI's phrasing guess at facts for you."
                    )
                
                st.subheader("Your Draft Plan of Action (Editable):")
                st.text_area(
                    "Review and edit your draft below:",
                    value=st.session_state.poa_draft,
                    height=400
                )
                
                st.success("Draft ready! Copy the text above to submit via Seller Central.")

    st.divider()
    st.caption(DISCLAIMER_TEXT)

if __name__ == "__main__":
    main()