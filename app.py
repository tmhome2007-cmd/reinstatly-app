"""
Reinstatly - Amazon POA Builder
--------------------------------
FEATURES:
1. Developer Debug View (accessible via secret URL query param).
2. Strict zero-fabrication guardrails with automatic placeholder sanitation.
3. Verification / Account Integrity category detection with specialist warnings.
4. Closed Appeal Channel / Final Decision detection layer with warning banners.
5. Robust English language detection using word boundaries.
6. Comprehensive integration of user-provided facts across all input fields.
7. Dynamic active model fetching to handle Groq API deprecations smoothly.
"""

import streamlit as st
import json
import hashlib
import re
from datetime import datetime
import pandas as pd
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# CONSTANTEN & INITIALISATIE
# ==========================================
FALLBACK_MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.2
LOG_FILE = "sessions_log.json"
DEBUG_SECRET_TOKEN = "MijnGeheimeSleutel123"

DISCLAIMER_TEXT = (
    "⚠️ **Disclaimer:** This tool drafts a starting point based only on what you provide. "
    "It is not legal advice, does not guarantee reinstatement, and does not replace "
    "professional review for complex cases (IP disputes, related-account issues, or anything involving legal risk)."
)

SPECIALIST_WARNING = (
    "🚨 **Critical Notice for Account Integrity / Verification Suspensions:**\n"
    "This category often involves P-4 identity checks, video verification, or suspected account linkage/Section 3 flags. "
    "A written Plan of Action alone frequently fails to resolve these cases unless the underlying identity mismatch or linked account issue is identified and resolved first. "
    "Strongly consider consulting an account reinstatement specialist or legal counsel before submitting an appeal."
)

FINAL_DECISION_WARNING = (
    "⚠️ **This notice indicates Amazon has closed this specific appeal channel.**\n"
    "Language like 'final decision' or 'unlikely to change the outcome' means a standard resubmission is unlikely to be reviewed. "
    "If you have NEW evidence not previously submitted, emphasize what is specifically new. "
    "Otherwise, realistic next steps typically include: escalating through Account Health Support and explicitly requesting manual review, "
    "consulting a specialist about arbitration or legal options, or accepting this outcome. "
    "A generic Plan of Action alone is unlikely to reopen a case Amazon has explicitly closed."
)

@st.cache_resource
def get_working_model_name():
    """Haalt de actieve modellijst op bij Groq en kiest een beschikbaar model."""
    if "GROQ_API_KEY" not in st.secrets:
        return FALLBACK_MODEL_NAME
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        models_list = client.models.list()
        available_models = [m.id for m in models_list.data]
        
        # Probeer eerst een GPT-OSS, Qwen of Llama model te selecteren
        for model in available_models:
            if any(k in model.lower() for k in ["gpt-oss", "qwen", "llama"]):
                return model
                
        if available_models:
            return available_models[0]
            
    except Exception:
        pass
        
    return FALLBACK_MODEL_NAME

@st.cache_resource
def get_llm(model_override=None):
    """Initialiseert de ChatGroq LLM met een dynamisch goedgekeurd model."""
    if "GROQ_API_KEY" not in st.secrets:
        return None
    
    selected_model = model_override if model_override else get_working_model_name()
    
    return ChatGroq(
        model=selected_model, 
        temperature=TEMPERATURE,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

# ==========================================
# DEBUG VIEW (DEVELOPER ONLY)
# ==========================================

def render_debug_view():
    """Toont een verborgen debug-dashboard als de juiste URL-parameter is meegegeven."""
    if st.query_params.get("debug") == DEBUG_SECRET_TOKEN:
        st.title("🛠️ Developer Debug View - Sessions Log")
        st.warning("Je bevindt je in de afgeschermde ontwikkelaarsomgeving.")
        
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
                
            if logs:
                st.metric(label="Totaal aantal gelogde sessies", value=len(logs))
                df = pd.DataFrame(logs)
                
                expected_columns = {
                    "timestamp": "Onbekend",
                    "assigned_category": "Unclear",
                    "is_final_decision": False,
                    "notice_hash": "N/A"
                }
                
                for col, default_val in expected_columns.items():
                    if col not in df.columns:
                        df[col] = default_val
                
                df.fillna(value=expected_columns, inplace=True)
                
                df_display = df[["timestamp", "assigned_category", "is_final_decision", "notice_hash"]].copy()
                df_display.columns = ["Tijdstip", "Categorie", "Final Decision", "Notice Hash"]
                
                st.dataframe(df_display, use_container_width=True)
                
                json_string = json.dumps(logs, indent=4)
                st.download_button(
                    label="📥 Download sessions_log.json",
                    data=json_string,
                    file_name="sessions_log.json",
                    mime="application/json"
                )
            else:
                st.info("Het bestand `sessions_log.json` is nog leeg.")
                
        except FileNotFoundError:
            st.error("Het bestand `sessions_log.json` is nog niet aangemaakt.")
        except json.JSONDecodeError:
            st.error("Het bestand `sessions_log.json` bevat ongeldige JSON-data.")
            
        st.divider()

# ==========================================
# HULPFUNCTIES & GUARDRAILS
# ==========================================

def is_english(text: str) -> bool:
    """Checkt op niet-Engelse stopwoorden om vreemde talen te blokkeren."""
    text_lower = text.lower()
    non_english_patterns = [
        r'\bhet\b', r'\bhetzelf\b', r'\bactieplan\b', r'\bgeschorst\b', 
        r'\bbeste\b', r'\bverkoper\b', r'\bbeleid\b', r'\bgelieve\b', 
        r'\bingediend\b', r'\bnicht\b', r'\bvotre\b', r'\bcompte\b'
    ]
    matches = sum(1 for pattern in non_english_patterns if re.search(pattern, text_lower))
    return matches < 2

def log_session(notice_text: str, category: str, is_final: bool):
    """Slaat geanonimiseerde sessielogs op."""
    notice_hash = hashlib.sha256(notice_text.encode('utf-8')).hexdigest()
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "notice_hash": notice_hash,
        "assigned_category": category,
        "is_final_decision": is_final
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
    """Vervangt gefabriceerde details door placeholders als ze niet in de input stonden."""
    user_input_lower = combined_user_inputs.lower()
    sanitized_draft = draft_text
    replacements_made = False

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
        for match in re.findall(pattern, sanitized_draft, flags=re.IGNORECASE):
            if match.lower() not in user_input_lower:
                sanitized_draft = re.sub(re.escape(match), "[insert exact quantity of units]", sanitized_draft, flags=re.IGNORECASE)
                replacements_made = True

    sourcing_keywords = [
        r'\b(?:an?\s+)?online\s+liquidation\s+(?:website|source|retailer)\b',
        r'\bliquidation\s+website\b',
        r'\bunauthorized\s+(?:distributor|wholesaler|supplier)\b',
        r'\bthird-party\s+liquidator\b',
        r'\bretail\s+arbitrage\s+source\b'
    ]
    for pattern in sourcing_keywords:
        for match in re.findall(pattern, sanitized_draft, flags=re.IGNORECASE):
            if match.lower() not in user_input_lower:
                sanitized_draft = re.sub(re.escape(match), "[insert your actual product source/supplier]", sanitized_draft, flags=re.IGNORECASE)
                replacements_made = True

    date_pattern = r'\b(?:\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?)\b'
    for date_str in re.findall(date_pattern, sanitized_draft, flags=re.IGNORECASE):
        if date_str.lower() not in user_input_lower:
            sanitized_draft = re.sub(re.escape(date_str), "[insert date]", sanitized_draft, flags=re.IGNORECASE)
            replacements_made = True

    return sanitized_draft, replacements_made

# ==========================================
# LLM CALLS MET AUTOMATISCHE FALLBACK
# ==========================================

def run_diagnosis(notice_text: str, seller_type: str, user_cause: str, user_actions: str, occurrence: str, account_health: str):
    """Analyseert het schorsingsbericht met behulp van Groq Cloud AI."""
    system_prompt = (
        "You are an expert Amazon account reinstatement analyst.\n"
        "STRICT RULE: ONLY classify the root cause based on the literal text of the notice provided. "
        "Never assume facts, dates, product categories, or history not explicitly stated by the user.\n\n"
        "Allowed Categories:\n"
        "1. Verification / Account Integrity (P-4, fraud, or illegal activity flag)\n"
        "2. Inauthenticity/IP complaint\n"
        "3. Order Defect Rate / performance metrics\n"
        "4. Related account\n"
        "5. Restricted category/listing policy\n"
        "6. Review manipulation\n"
        "7. Other/Unclear\n\n"
        "FINAL DECISION DETECTION:\n"
        "Check if the notice contains language indicating Amazon has closed the appeal channel, such as:\n"
        "- 'final decision'\n"
        "- 'unlikely to change the outcome'\n"
        "- 'may not respond to further emails'\n"
        "- 'evaluation of your account is complete'\n"
        "- 'will not be reactivated'\n\n"
        "Output Format EXACTLY as follows:\n"
        "Category: [Selected Category]\n"
        "Final Decision Flag: [Yes or No]\n"
        "Explanation: [2-3 sentence plain-English explanation quoting ONLY phrases from the notice text]."
    )
    
    user_message = (
        f"Notice Text:\n{notice_text}\n\n"
        f"Seller Type: {seller_type}\n"
        f"Occurrence History: {occurrence}\n"
        f"Account Health Rating: {account_health if account_health else 'Not provided'}\n"
        f"User Supposed Cause: {user_cause if user_cause else 'None provided'}\n"
        f"Actions Taken: {user_actions if user_actions else 'None provided'}"
    )
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    try:
        llm = get_llm()
        chain = prompt | llm
        response = chain.invoke({"input": user_message})
        return response.content
    except Exception:
        try:
            llm_fallback = get_llm(model_override=FALLBACK_MODEL_NAME)
            chain_fallback = prompt | llm_fallback
            response = chain_fallback.invoke({"input": user_message})
            return response.content
        except Exception as fallback_error:
            st.error(f"⚠️ Er is een fout opgetreden bij de AI-service: {fallback_error}")
            return None

def run_poa_generation(notice_text: str, seller_type: str, user_cause: str, user_actions: str, occurrence: str, account_health: str, diagnosis: str, is_final_decision: bool):
    """Genereert het Plan van Aanpak (POA)."""
    final_decision_instruction = ""
    if is_final_decision:
        final_decision_instruction = (
            "\nCRITICAL TONAL SHIFT - CLOSED APPEAL CHANNEL:\n"
            "- The notice indicates Amazon has issued a final decision or closed the standard appeal channel.\n"
            "- Frame the analysis around what, if anything, is genuinely NEW evidence.\n"
        )
    
    system_prompt = (
        "You are a strict Amazon Appeal Drafting Assistant.\n\n"
        "INTEGRATION OF USER FACTS:\n"
        "- Include all user-confirmed actions as specific, factual statements in Section 2.\n"
        "- Only use bracketed placeholders for information the user did NOT provide.\n"
        f"{final_decision_instruction}\n"
        "ABSOLUTE REQUIREMENT - ZERO NARRATIVE FABRICATION:\n"
        "- You must NEVER invent or assume story details, sourcing methods, unit counts, or dates.\n\n"
        "STRUCTURE:\n"
        "Generate a structured Plan of Action (POA) with EXACTLY three sections:\n"
        "1. Greater Detail on the Root Cause of the Issue\n"
        "2. Immediate Corrective Actions Taken\n"
        "3. Preventive Measures Implemented\n"
    )
    
    user_message = (
        f"Notice Text: {notice_text}\n"
        f"Seller Type: {seller_type}\n"
        f"Occurrence: {occurrence}\n"
        f"Account Health: {account_health if account_health else 'Not specified'}\n"
        f"Perceived Cause: {user_cause if user_cause else 'None provided'}\n"
        f"Actions Already Taken: {user_actions if user_actions else 'None provided'}\n"
        f"Confirmed Diagnosis: {diagnosis}\n"
        f"Is Final Decision: {'Yes' if is_final_decision else 'No'}"
    )
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    try:
        llm = get_llm()
        chain = prompt | llm
        response = chain.invoke({"input": user_message})
        return response.content
    except Exception:
        try:
            llm_fallback = get_llm(model_override=FALLBACK_MODEL_NAME)
            chain_fallback = prompt | llm_fallback
            response = chain_fallback.invoke({"input": user_message})
            return response.content
        except Exception as fallback_error:
            st.error(f"⚠️ Er is een fout opgetreden bij de AI-service: {fallback_error}")
            return None

# ==========================================
# STREAMLIT UI MAIN
# ==========================================

def main():
    st.set_page_config(page_title="Reinstatly - Amazon POA Builder", page_icon="📦", layout="centered")
    
    render_debug_view()

    st.title("📦 Reinstatly")
    st.caption("Privacy-First Amazon Suspension Plan of Action (POA) Generator")
    
    if "GROQ_API_KEY" not in st.secrets or not st.secrets["GROQ_API_KEY"]:
        st.error("❌ **GROQ_API_KEY not configured.** Please add your Groq API key to Streamlit secrets.")
        st.stop()

    st.info(DISCLAIMER_TEXT)
    st.divider()

    if "step" not in st.session_state:
        st.session_state.step = 1
    if "diagnosis" not in st.session_state:
        st.session_state.diagnosis = None
    if "is_final_decision" not in st.session_state:
        st.session_state.is_final_decision = False
    if "diagnosis_confirmed" not in st.session_state:
        st.session_state.diagnosis_confirmed = False
    if "poa_draft" not in st.session_state:
        st.session_state.poa_draft = None
    if "replacements_were_made" not in st.session_state:
        st.session_state.replacements_were_made = False

    # STEP 1: INTAKE
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

    occurrence = st.selectbox(
        "Is this the first time you've received this type of notice, or has it happened before?",
        ["First time", "Happened before", "Not sure"]
    )

    account_health = st.text_input(
        "Do you know your Account Health rating at the time of this notice? (optional)",
        placeholder="e.g. 200, Good, At Risk, or Not sure"
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
                diag_result = run_diagnosis(
                    notice_text, seller_type, user_cause, user_actions, occurrence, account_health
                )
                if diag_result:
                    st.session_state.diagnosis = diag_result
                    st.session_state.step = 2
                    st.session_state.diagnosis_confirmed = False
                    
                    is_final = bool(re.search(r'Final Decision Flag:\s*Yes', diag_result, re.IGNORECASE))
                    st.session_state.is_final_decision = is_final

                    cat_match = re.search(r'Category:\s*(.*?)(?=\n|$)', diag_result)
                    category_found = cat_match.group(1) if cat_match else "Unclear"
                    log_session(notice_text, category_found, is_final)
                    st.rerun()

    # STEP 2: ROOT CAUSE DIAGNOSIS
    if st.session_state.step >= 2 and st.session_state.diagnosis:
        st.divider()
        st.header("Step 2: Root Cause Diagnosis")
        
        st.subheader("AI Analysis Result:")
        st.write(st.session_state.diagnosis)
        
        if st.session_state.is_final_decision:
            st.warning(FINAL_DECISION_WARNING)

        if "Verification / Account Integrity" in st.session_state.diagnosis:
            st.warning(SPECIALIST_WARNING)

        st.write("---")
        st.write("**Does this diagnosis match your situation?**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, continue to POA Draft"):
                st.session_state.diagnosis_confirmed = True
                st.session_state.step = 3
                st.session_state.poa_draft = None
                st.rerun()
                
        with col2:
            if st.button("❌ No, let me re-describe it"):
                st.session_state.diagnosis_confirmed = False
                st.session_state.step = 1
                st.info("Please update the details in Step 1 and run the analysis again.")

    # STEP 3: PLAN OF ACTION DRAFT
    if st.session_state.step == 3:
        st.divider()
        st.header("Step 3: Plan of Action Draft")
        
        if st.session_state.is_final_decision:
            st.warning(FINAL_DECISION_WARNING)

        if "Verification / Account Integrity" in st.session_state.diagnosis:
            st.error(SPECIALIST_WARNING)

        if not st.session_state.diagnosis_confirmed:
            st.warning("Please confirm the diagnosis in Step 2 before generating the draft.")
        else:
            if not st.session_state.poa_draft:
                with st.spinner("Generating structured Plan of Action..."):
                    raw_draft = run_poa_generation(
                        notice_text, seller_type, user_cause, user_actions, occurrence, account_health, 
                        st.session_state.diagnosis, st.session_state.is_final_decision
                    )
                    
                    if raw_draft:
                        combined_inputs = f"{notice_text} {user_cause} {user_actions} {occurrence} {account_health}"
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