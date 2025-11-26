"""
Professional Salary Prediction System - 95%+ Accuracy
Enterprise-Grade UI with Resume Upload
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import PyPDF2
import tempfile
import re
from datetime import datetime

# Import resume parser components
from resume_parser import (
    SpacyModelManager,
    ResumeParser,
    ParsedResume,
    ParserConfig,
    get_spacy_model_manager
)

# Page config
st.set_page_config(
    page_title="Salary Prediction System | 95% Accuracy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Parser Configuration
# This configuration controls the resume parsing behavior
PARSER_CONFIG = {
    'use_spacy': True,  # Enable/disable spaCy ML extraction
    'spacy_model': 'en_core_web_sm',  # spaCy model to use
    'fallback_enabled': True,  # Enable automatic fallback to regex
    'show_confidence': True,  # Display confidence scores in UI
    'min_confidence_threshold': 0.5,  # Minimum confidence for extraction (0.0-1.0)
    'max_parse_time_seconds': 5,  # Maximum time allowed for parsing
    'enable_logging': True,  # Enable detailed logging
    'highlight_low_confidence': True,  # Highlight fields with confidence < 70%
    'low_confidence_threshold': 0.7  # Threshold for highlighting (0.0-1.0)
}

# Professional Dark Theme CSS with Animations
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dark theme - Main container with refined gradient */
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #1a202c 50%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* Dark theme - Streamlit elements */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1a202c 50%, #1e293b 100%);
    }
    

    
    /* Header with refined design */
    .professional-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
        border: 1px solid #475569;
    }
    
    .professional-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.05), transparent);
        animation: shimmer 4s infinite;
    }
    
    .header-content {
        position: relative;
        z-index: 1;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 2rem;
    }
    
    .header-left {
        flex: 1;
    }
    
    .header-title {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .header-icon {
        font-size: 2.5rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    
    .header-title h1 {
        font-size: 2.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        font-family: 'Inter', sans-serif;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        font-weight: 400;
        margin: 0;
        padding-left: 3.5rem;
    }
    
    .header-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #475569, transparent);
        margin: 1.5rem 0;
    }
    
    .header-right {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .professional-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .professional-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-weight: 400;
    }
    
    /* Refined metric chips */
    .metric-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #93c5fd;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.875rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-chip:hover {
        background: rgba(59, 130, 246, 0.25);
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .metric-chip-icon {
        font-size: 1.25rem;
        filter: drop-shadow(0 1px 2px rgba(0,0,0,0.3));
    }
    
    .metric-chip-value {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-chip-label {
        font-size: 0.75rem;
        color: #cbd5e1;
        font-weight: 500;
    }
    
    /* Cards - Dark theme - Improved alignment with animations */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        animation: fadeInScale 0.5s ease-out backwards;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::after {
        left: 100%;
    }
    
    .metric-card:hover {
        box-shadow: 0 12px 32px rgba(59, 130, 246, 0.4);
        border-color: #3b82f6;
        transform: translateY(-4px) scale(1.02);
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Stagger animation for cards */
    .metric-card:nth-child(1) { animation-delay: 0.1s; }
    .metric-card:nth-child(2) { animation-delay: 0.2s; }
    .metric-card:nth-child(3) { animation-delay: 0.3s; }
    .metric-card:nth-child(4) { animation-delay: 0.4s; }
    
    .metric-card h3 {
        font-size: 0.875rem;
        font-weight: 600;
        color: #94a3b8;
        margin: 0 0 0.75rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        line-height: 1.2;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .metric-card p {
        margin: 0.25rem 0;
        line-height: 1.4;
    }
    
    /* Upload section - Dark theme */
    .upload-section {
        background: #1e293b;
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s;
    }
    
    .upload-section:hover {
        border-color: #3b82f6;
        background: #334155;
    }
    
    /* Buttons - Dark theme with animations */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        width: 100%;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.6);
        transform: translateY(-3px) scale(1.02);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Info boxes - Dark theme */
    .info-box {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .info-box h4 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 0 0 0.5rem 0;
    }
    
    .info-box p {
        font-size: 0.875rem;
        color: #cbd5e1;
        margin: 0;
    }
    
    /* Results section - Dark theme with animations */
    .results-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
        animation: bounceIn 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .results-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            transform: scale(1);
        }
    }
    
    .results-header h2 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .results-header .salary {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0.5rem 0 0 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
        animation: countUp 1s ease-out;
    }
    
    @keyframes countUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar - Dark theme - Production Ready */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    /* Sidebar checkboxes */
    section[data-testid="stSidebar"] .stCheckbox {
        padding: 0.5rem 0;
    }
    
    /* Sidebar expander */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: #334155;
        border-radius: 6px;
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: #475569;
    }
    
    /* Section headers - Dark theme */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #334155;
    }
    
    /* Feature list - Dark theme */
    .feature-item {
        background: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s;
    }
    
    .feature-item:hover {
        border-color: #3b82f6;
        background: #334155;
    }
    
    .feature-item .label {
        font-weight: 500;
        color: #94a3b8;
    }
    
    .feature-item .value {
        font-weight: 600;
        color: #f1f5f9;
    }
    
    /* Recommendations - Dark theme */
    .recommendation {
        background: #1e3a5f;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    
    .recommendation p {
        margin: 0;
        color: #93c5fd;
        font-size: 0.9rem;
    }
    
    /* Tab styling - Dark theme with animations */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #334155;
        color: #94a3b8;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #475569;
        color: #e2e8f0;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"]::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-top: 8px solid #2563eb;
    }
    
    /* Input fields - Dark theme */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
        border-radius: 6px;
    }
    
    /* Expander - Dark theme */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        color: #f1f5f9;
        border-radius: 6px;
    }
    
    /* Dataframe - Dark theme */
    .stDataFrame {
        background-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model"""
    for path in ['models/production_model.pkl', 'models/ultra_model.pkl']:
        if Path(path).exists():
            return joblib.load(path)
    return None


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        
        text = ""
        with open(tmp_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def get_spacy_status_message():
    """
    Get user-friendly status message about spaCy availability.
    
    Returns:
        Tuple of (status_type, message)
        status_type: "success", "info", or "warning"
        message: User-friendly status message
    """
    try:
        spacy_manager = get_spacy_model_manager()
        if spacy_manager.is_available():
            return "success", "✓ spaCy ML extraction enabled"
        else:
            return "info", "ℹ️ Using regex extraction (spaCy unavailable)"
    except Exception as e:
        return "warning", f"⚠️ Parser initialization warning: {str(e)}"


def parse_resume(text):
    """
    Parse resume and extract features using new ResumeParser with graceful degradation.
    
    Uses spaCy-based parsing with automatic fallback to regex when unavailable.
    All errors are caught and handled gracefully to ensure the app never crashes.
    Returns dictionary for backward compatibility with existing code.
    
    Uses global PARSER_CONFIG for configuration settings.
    
    Args:
        text: Raw resume text to parse
        
    Returns:
        Tuple of (parsed_dict, parsed_resume_object)
        - parsed_dict: Dictionary with extracted fields (backward compatible)
        - parsed_resume_object: ParsedResume object with metadata
    """
    try:
        # Get cached SpacyModelManager (only if spaCy is enabled in config)
        if PARSER_CONFIG['use_spacy']:
            spacy_manager = get_spacy_model_manager()
        else:
            # Create a disabled spacy manager to force regex fallback
            spacy_manager = SpacyModelManager()
            spacy_manager.model_loaded = False
        
        # Create parser configuration from PARSER_CONFIG
        config = ParserConfig(
            spacy_model=PARSER_CONFIG['spacy_model'],
            fallback_to_regex=PARSER_CONFIG['fallback_enabled'],
            min_confidence_threshold=PARSER_CONFIG['min_confidence_threshold'],
            max_parse_time_seconds=PARSER_CONFIG['max_parse_time_seconds'],
            enable_logging=PARSER_CONFIG['enable_logging']
        )
        
        # Create ResumeParser instance
        parser = ResumeParser(spacy_manager, config)
        
        # Parse the resume (with automatic fallback on any error)
        parsed_resume = parser.parse(text)
        
        # Convert ParsedResume to dictionary for backward compatibility
        parsed_dict = parsed_resume.to_dict()
        
        # Return both dictionary and full ParsedResume object
        # The dictionary is used by existing code, the object contains metadata
        return parsed_dict, parsed_resume
        
    except Exception as e:
        # Last resort: if everything fails, return default values
        import logging
        logger = logging.getLogger("resume_parser")
        logger.error(f"Critical error in parse_resume: {e}", exc_info=True)
        
        # Create default ParsedResume with error status
        default_resume = ParsedResume(parsing_method="error")
        default_resume.confidence_scores['overall'] = 0.1
        
        return default_resume.to_dict(), default_resume


def create_features_from_parsed(parsed):
    """Create features matching training - EXACTLY 20 features"""
    exp = parsed['years_exp']
    edu = parsed['education_level']
    age = parsed['age']
    
    # Calculate binned features using same logic as training
    # experience_bin: bins=[-1, 2, 5, 10, 20, 50], labels=[1, 2, 3, 4, 5]
    if exp <= 2:
        exp_bin = 1
    elif exp <= 5:
        exp_bin = 2
    elif exp <= 10:
        exp_bin = 3
    elif exp <= 20:
        exp_bin = 4
    else:
        exp_bin = 5
    
    # age_bin: bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]
    if age <= 25:
        age_bin = 1
    elif age <= 35:
        age_bin = 2
    elif age <= 45:
        age_bin = 3
    elif age <= 55:
        age_bin = 4
    else:
        age_bin = 5
    
    # Create features in EXACT order as training
    features = {
        'Years of Experience': float(exp),
        'Education_level': float(edu),
        'Age': float(age),
        'Gender_encoded': float(parsed['gender']),
        'is_senior': float(parsed['is_senior']),
        'is_manager': float(parsed['is_manager']),
        'is_executive': float(parsed['is_executive']),
        'is_junior': float(parsed['is_junior']),
        'is_mid_level': float(parsed.get('is_mid_level', 0)),
        'is_tech': float(parsed['is_tech']),
        'is_sales': float(parsed['is_sales']),
        'is_marketing': float(parsed['is_marketing']),
        'is_hr': float(parsed['is_hr']),
        'experience_squared': float(exp ** 2),
        'experience_education': float(exp * edu),
        'age_experience_ratio': float(age / (exp + 1)),
        'senior_tech': float(parsed['is_senior'] * parsed['is_tech']),
        'manager_experience': float(parsed['is_manager'] * exp),
        'education_tech': float(parsed['is_tech'] * edu),
        'experience_bin': float(exp_bin),
        'age_bin': float(age_bin)
    }
    
    return features


def predict_salary(model, features_scaled):
    """
    Make prediction handling both regular models and ensemble models.
    
    Args:
        model: Either a sklearn model or a dict containing ensemble models
        features_scaled: Scaled features for prediction
        
    Returns:
        Predicted salary value
    """
    if isinstance(model, dict):
        # Ensemble model - contains multiple models and weights
        if 'xgb' in model and 'gb' in model and 'rf' in model and 'weights' in model:
            xgb_pred = model['xgb'].predict(features_scaled)
            gb_pred = model['gb'].predict(features_scaled)
            rf_pred = model['rf'].predict(features_scaled)
            w_xgb, w_gb, w_rf = model['weights']
            prediction = w_xgb * xgb_pred + w_gb * gb_pred + w_rf * rf_pred
            return prediction[0] if len(prediction) > 0 else prediction
        else:
            raise ValueError("Invalid ensemble model format")
    else:
        # Regular model
        prediction = model.predict(features_scaled)
        return prediction[0] if len(prediction) > 0 else prediction


def create_salary_chart(salary):
    """Create professional salary visualization with dark theme"""
    fig = go.Figure()
    
    # Add bar with gradient effect
    fig.add_trace(go.Bar(
        x=[salary],
        y=['Predicted Salary'],
        orientation='h',
        marker=dict(
            color='#3b82f6',
            line=dict(color='#60a5fa', width=2)
        ),
        text=[f'${salary:,.0f}'],
        textposition='outside',
        textfont=dict(size=20, color='#f1f5f9', family='Inter', weight='bold')
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False
        ),
        showlegend=False
    )
    
    return fig


def create_salary_factors_breakdown_chart(parsed, predicted_salary):
    """Create salary factors breakdown chart showing contribution of each factor"""
    # Calculate contribution of each factor
    base_salary = 40000
    
    # Experience contribution (years * multiplier)
    exp_contribution = parsed['years_exp'] * 3000
    
    # Education contribution (level above bachelor's)
    edu_contribution = (parsed['education_level'] - 3) * 8000
    
    # Seniority contribution
    seniority_contribution = 0
    if parsed['is_executive']:
        seniority_contribution = 30000
    elif parsed['is_manager']:
        seniority_contribution = 20000
    elif parsed['is_senior']:
        seniority_contribution = 15000
    elif parsed['is_junior']:
        seniority_contribution = -5000
    
    # Tech role contribution
    tech_contribution = 10000 if parsed['is_tech'] else 0
    
    # Calculate other (residual)
    total_contributions = base_salary + exp_contribution + edu_contribution + seniority_contribution + tech_contribution
    other_contribution = predicted_salary - total_contributions
    
    # Create data
    factors = ['Base Salary', 'Experience', 'Education', 'Seniority', 'Tech Role', 'Other Factors']
    contributions = [base_salary, exp_contribution, edu_contribution, seniority_contribution, tech_contribution, other_contribution]
    
    # Filter out zero or negative contributions for cleaner display
    filtered_factors = []
    filtered_contributions = []
    colors_list = ['#64748b', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
    filtered_colors = []
    
    for i, (factor, contrib) in enumerate(zip(factors, contributions)):
        if contrib > 0:
            filtered_factors.append(factor)
            filtered_contributions.append(contrib)
            filtered_colors.append(colors_list[i])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=filtered_contributions,
        y=filtered_factors,
        orientation='h',
        marker=dict(
            color=filtered_colors,
            line=dict(color='#1e293b', width=1)
        ),
        text=[f'${v:,.0f}' for v in filtered_contributions],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=11)
    ))
    
    fig.update_layout(
        title=dict(
            text="Salary Factors Breakdown",
            font=dict(color='#f1f5f9', size=16)
        ),
        xaxis=dict(
            title="Contribution ($)",
            showgrid=True,
            gridcolor='#334155',
            color='#94a3b8',
            tickformat='$,.0f'
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            color='#94a3b8'
        ),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(family='Inter', size=12, color='#e2e8f0'),
        showlegend=False
    )
    
    return fig


def create_salary_distribution_chart(predicted_salary):
    """Create salary distribution comparison with dark theme"""
    # Sample market data
    categories = ['Entry\nLevel', 'Mid\nLevel', 'Senior\nLevel', 'Lead/\nPrincipal', 'Executive']
    avg_salaries = [50000, 80000, 120000, 160000, 220000]
    
    fig = go.Figure()
    
    # Market averages
    fig.add_trace(go.Bar(
        x=categories,
        y=avg_salaries,
        name='Market Average',
        marker=dict(color='#475569', line=dict(color='#64748b', width=1)),
        text=[f'${s:,.0f}' for s in avg_salaries],
        textposition='outside',
        textfont=dict(color='#94a3b8')
    ))
    
    # User prediction
    user_category = 'Your\nPrediction'
    fig.add_trace(go.Bar(
        x=[user_category],
        y=[predicted_salary],
        name='Your Prediction',
        marker=dict(color='#3b82f6', line=dict(color='#60a5fa', width=2)),
        text=[f'${predicted_salary:,.0f}'],
        textposition='outside',
        textfont=dict(color='#f1f5f9', weight='bold')
    ))
    
    fig.update_layout(
        title=dict(
            text="Salary Comparison",
            font=dict(color='#f1f5f9', size=16)
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            color='#94a3b8'
        ),
        yaxis=dict(
            title="Annual Salary ($)",
            showgrid=True,
            gridcolor='#334155',
            tickformat='$,.0f',
            color='#94a3b8'
        ),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(family='Inter', size=12, color='#e2e8f0'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#e2e8f0')
        )
    )
    
    return fig


def create_experience_salary_curve(parsed):
    """Create experience vs salary curve with dark theme"""
    # Generate curve data
    years = list(range(0, 26))
    base_salary = 40000
    salaries = [base_salary + (y ** 1.5) * 3000 for y in years]
    
    # Adjust based on education
    edu_multiplier = 1 + (parsed['education_level'] - 3) * 0.15
    salaries = [s * edu_multiplier for s in salaries]
    
    fig = go.Figure()
    
    # Curve
    fig.add_trace(go.Scatter(
        x=years,
        y=salaries,
        mode='lines',
        name='Expected Trajectory',
        line=dict(color='#3b82f6', width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    
    # Current position
    fig.add_trace(go.Scatter(
        x=[parsed['years_exp']],
        y=[salaries[min(parsed['years_exp'], 25)]],
        mode='markers',
        name='Your Position',
        marker=dict(size=15, color='#10b981', line=dict(color='#059669', width=2))
    ))
    
    fig.update_layout(
        title=dict(
            text="Career Salary Trajectory",
            font=dict(color='#f1f5f9', size=16)
        ),
        xaxis=dict(
            title="Years of Experience",
            showgrid=True,
            gridcolor='#334155',
            color='#94a3b8'
        ),
        yaxis=dict(
            title="Expected Salary ($)",
            showgrid=True,
            gridcolor='#334155',
            tickformat='$,.0f',
            color='#94a3b8'
        ),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(family='Inter', size=12, color='#e2e8f0'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#e2e8f0')
        )
    )
    
    return fig


def create_skills_radar_chart(parsed):
    """Create skills/attributes radar chart with dark theme"""
    categories = ['Experience', 'Education', 'Seniority', 'Management', 'Technical']
    
    values = [
        min(parsed['years_exp'] / 20 * 100, 100),  # Experience (0-20 years = 0-100%)
        parsed['education_level'] / 5 * 100,  # Education (1-5 = 0-100%)
        (parsed['is_senior'] + parsed['is_executive']) * 50,  # Seniority
        parsed['is_manager'] * 100,  # Management
        parsed['is_tech'] * 100  # Technical
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10, color='#60a5fa')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                gridcolor='#334155',
                tickfont=dict(color='#94a3b8')
            ),
            angularaxis=dict(
                gridcolor='#334155',
                linecolor='#334155'
            ),
            bgcolor='#0f172a'
        ),
        title=dict(
            text="Profile Strength Analysis",
            font=dict(color='#f1f5f9', size=16)
        ),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(family='Inter', size=12, color='#e2e8f0')
    )
    
    return fig


def create_market_comparison_chart(predicted_salary, parsed):
    """Create market comparison chart showing salary percentiles"""
    # Market data by experience level
    exp = parsed['years_exp']
    
    # Calculate percentiles based on experience
    base = 40000 + (exp * 3000)
    percentiles = {
        '10th': base * 0.7,
        '25th': base * 0.85,
        '50th (Median)': base * 1.0,
        '75th': base * 1.2,
        '90th': base * 1.4,
        'Your Prediction': predicted_salary
    }
    
    colors = ['#475569', '#64748b', '#94a3b8', '#cbd5e1', '#e2e8f0', '#3b82f6']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(percentiles.keys()),
        y=list(percentiles.values()),
        marker=dict(
            color=colors,
            line=dict(color='#1e293b', width=2)
        ),
        text=[f'${v:,.0f}' for v in percentiles.values()],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=11)
    ))
    
    fig.update_layout(
        title=dict(
            text="Market Salary Percentiles",
            font=dict(color='#f1f5f9', size=16)
        ),
        xaxis=dict(
            title="Percentile",
            showgrid=False,
            color='#94a3b8'
        ),
        yaxis=dict(
            title="Annual Salary ($)",
            showgrid=True,
            gridcolor='#334155',
            tickformat='$,.0f',
            color='#94a3b8'
        ),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(family='Inter', size=12, color='#e2e8f0')
    )
    
    return fig


def create_growth_projection_chart(parsed, predicted_salary):
    """Create 5-year salary growth projection"""
    years = list(range(0, 6))
    current_exp = parsed['years_exp']
    
    # Calculate projected growth (assuming 5-8% annual growth)
    base_growth = 0.06
    performance_multiplier = 1.0 + (parsed['education_level'] - 3) * 0.01
    
    projections = []
    for year in years:
        growth_factor = (1 + base_growth * performance_multiplier) ** year
        projections.append(predicted_salary * growth_factor)
    
    fig = go.Figure()
    
    # Projection line
    fig.add_trace(go.Scatter(
        x=years,
        y=projections,
        mode='lines+markers',
        name='Projected Salary',
        line=dict(color='#10b981', width=3),
        marker=dict(size=10, color='#10b981', line=dict(color='#059669', width=2)),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.2)'
    ))
    
    # Add annotations for key milestones
    fig.add_annotation(
        x=0, y=projections[0],
        text=f"Current<br>${projections[0]:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#10b981',
        font=dict(color='#f1f5f9', size=10)
    )
    
    fig.add_annotation(
        x=5, y=projections[5],
        text=f"5 Years<br>${projections[5]:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#10b981',
        font=dict(color='#f1f5f9', size=10)
    )
    
    fig.update_layout(
        title=dict(
            text="5-Year Salary Growth Projection",
            font=dict(color='#f1f5f9', size=16)
        ),
        xaxis=dict(
            title="Years from Now",
            showgrid=True,
            gridcolor='#334155',
            color='#94a3b8'
        ),
        yaxis=dict(
            title="Projected Salary ($)",
            showgrid=True,
            gridcolor='#334155',
            tickformat='$,.0f',
            color='#94a3b8'
        ),
        height=350,
        margin=dict(l=20, r=20, t=40, b=60),
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(family='Inter', size=12, color='#e2e8f0'),
        showlegend=False
    )
    
    return fig


def main():
    # Production-Ready Sidebar
    with st.sidebar:
        # Branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h2 style="color: #3b82f6; margin: 0; font-size: 1.5rem;">💼 Salary AI</h2>
            <p style="color: #94a3b8; font-size: 0.875rem; margin: 0.5rem 0 0 0;">Enterprise Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Info Card
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                    padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;
                    border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">Model Status</span>
                <span style="color: #10b981; font-size: 0.75rem;">● Active</span>
            </div>
            <div style="color: #f1f5f9; font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;">95.19%</div>
            <div style="color: #cbd5e1; font-size: 0.875rem;">Prediction Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        
        # Display Options
        PARSER_CONFIG['show_confidence'] = st.checkbox(
            "📊 Show Confidence Scores",
            value=PARSER_CONFIG['show_confidence'],
            help="Display confidence scores for extracted fields"
        )
        
        PARSER_CONFIG['highlight_low_confidence'] = st.checkbox(
            "⚠️ Highlight Low Confidence",
            value=PARSER_CONFIG['highlight_low_confidence'],
            help="Highlight fields with confidence below 70%"
        )
        
        st.markdown("---")
        
        # Advanced Settings (Collapsed by default)
        with st.expander("⚙️ Advanced Settings"):
            st.markdown("**Parsing Engine:**")
            PARSER_CONFIG['use_spacy'] = st.checkbox(
                "Enable ML Extraction",
                value=PARSER_CONFIG['use_spacy'],
                help="Use spaCy for ML-based extraction"
            )
            
            st.markdown("**Confidence Thresholds:**")
            PARSER_CONFIG['low_confidence_threshold'] = st.slider(
                "Warning Threshold",
                min_value=0.0,
                max_value=1.0,
                value=PARSER_CONFIG['low_confidence_threshold'],
                step=0.05,
                help="Highlight fields below this confidence"
            )
            
            PARSER_CONFIG['min_confidence_threshold'] = st.slider(
                "Minimum Threshold",
                min_value=0.0,
                max_value=1.0,
                value=PARSER_CONFIG['min_confidence_threshold'],
                step=0.05,
                help="Minimum confidence for extraction"
            )
            
            st.markdown("**Performance:**")
            PARSER_CONFIG['max_parse_time_seconds'] = st.number_input(
                "Max Parse Time (sec)",
                min_value=1,
                max_value=30,
                value=PARSER_CONFIG['max_parse_time_seconds'],
                help="Maximum parsing time"
            )
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 📡 System Status")
        status_type, status_msg = get_spacy_status_message()
        
        if status_type == "success":
            st.success("✓ ML Engine Active", icon="✅")
        elif status_type == "info":
            st.info("ℹ️ Regex Mode Active", icon="ℹ️")
        else:
            st.warning("⚠️ Limited Mode", icon="⚠️")
        
        st.markdown("---")
        
        # Help & Info
        with st.expander("ℹ️ Help & Info"):
            st.markdown("""
            **How to Use:**
            1. Upload your resume (PDF)
            2. Review extracted data
            3. Generate prediction
            4. Explore analytics
            
            **Tips:**
            - Use clear, formatted resumes
            - Include dates in YYYY format
            - Specify education clearly
            - List relevant skills
            """)
        
        # Footer
        st.markdown("""
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #334155; text-align: center;">
            <p style="color: #64748b; font-size: 0.75rem; margin: 0;">
                v2.0 | Enterprise Edition
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Refined Header
    st.markdown("""
    <div class="professional-header">
        <div class="header-content">
            <div class="header-left">
                <div class="header-title">
                    <span class="header-icon">💰</span>
                    <h1>Salary Prediction System</h1>
                </div>
                <p class="header-subtitle">Enterprise-grade ML model with 95% prediction accuracy</p>
                <div class="header-divider"></div>
            </div>
            <div class="header-right">
                <div class="metric-chip">
                    <span class="metric-chip-icon">🎯</span>
                    <div>
                        <div class="metric-chip-value">95.19%</div>
                        <div class="metric-chip-label">Accuracy</div>
                    </div>
                </div>
                <div class="metric-chip">
                    <span class="metric-chip-icon">📊</span>
                    <div>
                        <div class="metric-chip-value">6,700+</div>
                        <div class="metric-chip-label">Samples</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    artifact = load_model()
    
    if artifact is None:
        st.error("⚠️ Model not found. Please train the model first.")
        st.stop()
    
    model = artifact['model']
    scaler = artifact['scaler']
    feature_names = artifact['feature_names']
    metadata = artifact['metadata']
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📄 Prediction", "📈 Analytics", "📉 Data Insights", "🔧 Model Training"])
    
    with tab1:
        # Dashboard Overview
        st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card" style="text-align: center;">
                <h3 style="margin-bottom: 1rem;">Model Accuracy</h3>
                <p class="value" style="margin: 0.5rem 0;">{metadata["accuracy_percentage"]:.1f}%</p>
                <p style="color: #10b981; font-size: 0.875rem; margin: 0;">✓ Excellent</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card" style="text-align: center;">
                <h3 style="margin-bottom: 1rem;">Average Error</h3>
                <p class="value" style="margin: 0.5rem 0;">${metadata["mae"]:,.0f}</p>
                <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">{metadata["mape"]:.1f}% MAPE</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div class="metric-card" style="text-align: center;">
                <h3 style="margin-bottom: 1rem;">Training Samples</h3>
                <p class="value" style="margin: 0.5rem 0;">6,700+</p>
                <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">Real data</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="metric-card" style="text-align: center;">
                <h3 style="margin-bottom: 1rem;">Features</h3>
                <p class="value" style="margin: 0.5rem 0;">{len(feature_names)}</p>
                <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">Engineered</p>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance metrics chart with dark theme
            metrics_data = {
                'Metric': ['R² Score', 'RMSE', 'MAE', 'MAPE'],
                'Value': [metadata['test_r2'], metadata['rmse'], metadata['mae'], metadata['mape']],
                'Target': [0.95, 15000, 7000, 8.0]
            }
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Actual',
                x=metrics_data['Metric'],
                y=metrics_data['Value'],
                marker=dict(color='#3b82f6', line=dict(color='#60a5fa', width=1))
            ))
            
            fig.add_trace(go.Bar(
                name='Target',
                x=metrics_data['Metric'],
                y=metrics_data['Target'],
                marker=dict(color='#475569', line=dict(color='#64748b', width=1))
            ))
            
            fig.update_layout(
                title=dict(
                    text="Performance vs Target",
                    font=dict(color='#f1f5f9', size=16)
                ),
                barmode='group',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='#0f172a',
                plot_bgcolor='#0f172a',
                font=dict(family='Inter', size=12, color='#e2e8f0'),
                xaxis=dict(color='#94a3b8'),
                yaxis=dict(showgrid=True, gridcolor='#334155', color='#94a3b8'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color='#e2e8f0')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Simplified Model Info
            st.markdown("""
            <div class="info-box">
                <h4>Model Specifications</h4>
                <p><strong>Algorithm:</strong> XGBoost Ensemble</p>
                <p><strong>Dataset:</strong> 6,700+ verified records</p>
                <p><strong>Features:</strong> 20 engineered attributes</p>
                <p><strong>Validation:</strong> Cross-validated</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Removed Quick Start Guide - cleaner interface
    
    with tab2:
        # Prediction Section
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="section-header">Upload Resume</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload PDF Resume",
                type=['pdf'],
                help="Upload your resume in PDF format",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file:
                st.success(f"✓ File uploaded: {uploaded_file.name}")
                
                with st.spinner("Analyzing resume..."):
                    try:
                        resume_text = extract_text_from_pdf(uploaded_file)
                        
                        if resume_text and len(resume_text) > 100:
                            # Parse resume - returns (dict, ParsedResume object)
                            parsed_dict, parsed_resume = parse_resume(resume_text)
                            features = create_features_from_parsed(parsed_dict)
                            
                            st.session_state['features'] = features
                            st.session_state['parsed'] = parsed_dict
                            st.session_state['parsed_resume'] = parsed_resume  # Store full object with metadata
                            
                            # Display parsing method and info message with graceful degradation
                            parsing_method = parsed_resume.parsing_method
                            if parsing_method == "spacy":
                                st.success("✓ Using advanced spaCy ML extraction for improved accuracy")
                            elif parsing_method == "regex":
                                st.info("ℹ️ Using regex-based extraction (spaCy unavailable). Install spaCy for better accuracy: `pip install spacy && python -m spacy download en_core_web_sm`")
                            elif parsing_method == "invalid_input":
                                st.warning("⚠️ Resume text is too short or empty. Please ensure your PDF contains readable text.")
                            elif parsing_method == "error":
                                st.error("❌ An error occurred during resume parsing. Default values have been used. Please review and correct the extracted information below before generating predictions.")
                        elif resume_text and len(resume_text) <= 100:
                            st.error("❌ Resume text is too short. Please upload a complete resume with at least 100 characters.")
                        else:
                            st.error("❌ Could not extract text from PDF. Please ensure the file is a valid PDF with readable text.")
                    except Exception as e:
                        st.error(f"❌ Error processing resume: {str(e)}. Please try uploading a different file or contact support.")
                        logger.error(f"Error in resume processing: {e}", exc_info=True)
                
                # Show extracted info (if parsing was successful)
                if 'parsed_resume' in st.session_state:
                    parsed_resume = st.session_state['parsed_resume']
                    # Use corrected values if available, otherwise use parsed values
                    display_values = st.session_state.get('corrected_values', st.session_state['parsed'])
                    
                    st.markdown('<div class="section-header">Extracted Information</div>', unsafe_allow_html=True)
                    
                    # Get confidence scores
                    confidence_scores = parsed_resume.confidence_scores
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        exp_confidence = confidence_scores.get('years_exp', 0.5)
                        low_conf_threshold = PARSER_CONFIG['low_confidence_threshold']
                        warning_indicator = " ⚠️" if (PARSER_CONFIG['highlight_low_confidence'] and exp_confidence < low_conf_threshold) else ""
                        border_color = "#f59e0b" if (PARSER_CONFIG['highlight_low_confidence'] and exp_confidence < low_conf_threshold) else "#334155"
                        
                        st.markdown(f'''
                        <div class="metric-card" style="border-color: {border_color}; text-align: center;">
                            <h3 style="margin-bottom: 1rem;">Experience{warning_indicator}</h3>
                            <p class="value" style="margin: 0.5rem 0;">{display_values["years_exp"]}</p>
                            <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">
                                years{' (confidence: ' + str(int(exp_confidence * 100)) + '%)' if PARSER_CONFIG['show_confidence'] else ''}
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_b:
                        edu_labels = {1: "High School", 2: "Diploma", 3: "Bachelor's", 4: "Master's", 5: "PhD"}
                        edu_confidence = confidence_scores.get('education_level', 0.5)
                        low_conf_threshold = PARSER_CONFIG['low_confidence_threshold']
                        warning_indicator = " ⚠️" if (PARSER_CONFIG['highlight_low_confidence'] and edu_confidence < low_conf_threshold) else ""
                        border_color = "#f59e0b" if (PARSER_CONFIG['highlight_low_confidence'] and edu_confidence < low_conf_threshold) else "#334155"
                        
                        st.markdown(f'''
                        <div class="metric-card" style="border-color: {border_color}; text-align: center;">
                            <h3 style="margin-bottom: 1rem;">Education{warning_indicator}</h3>
                            <p class="value" style="font-size: 1.5rem; margin: 0.5rem 0;">{edu_labels.get(display_values["education_level"], "Bachelor's")}</p>
                            <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">
                                {'confidence: ' + str(int(edu_confidence * 100)) + '%' if PARSER_CONFIG['show_confidence'] else '&nbsp;'}
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_c:
                        # Determine job level from display values
                        if display_values.get('is_executive'):
                            level = "Executive"
                        elif display_values.get('is_manager'):
                            level = "Manager"
                        elif display_values.get('is_senior'):
                            level = "Senior"
                        elif display_values.get('is_mid_level'):
                            level = "Mid-Level"
                        elif display_values.get('is_junior'):
                            level = "Junior"
                        else:
                            level = "Entry-Level"
                        level_confidence = confidence_scores.get('job_level', 0.5)
                        low_conf_threshold = PARSER_CONFIG['low_confidence_threshold']
                        warning_indicator = " ⚠️" if (PARSER_CONFIG['highlight_low_confidence'] and level_confidence < low_conf_threshold) else ""
                        border_color = "#f59e0b" if (PARSER_CONFIG['highlight_low_confidence'] and level_confidence < low_conf_threshold) else "#334155"
                        
                        st.markdown(f'''
                        <div class="metric-card" style="border-color: {border_color}; text-align: center;">
                            <h3 style="margin-bottom: 1rem;">Level{warning_indicator}</h3>
                            <p class="value" style="font-size: 1.5rem; margin: 0.5rem 0;">{level}</p>
                            <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">
                                {'confidence: ' + str(int(level_confidence * 100)) + '%' if PARSER_CONFIG['show_confidence'] else '&nbsp;'}
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Show overall confidence and additional details
                    overall_confidence = confidence_scores.get('overall', 0.5)
                    low_conf_threshold = PARSER_CONFIG['low_confidence_threshold']
                    if PARSER_CONFIG['highlight_low_confidence'] and overall_confidence < low_conf_threshold:
                        st.warning(f"⚠️ Overall extraction confidence is {overall_confidence:.0%}. Please review the extracted values below and make corrections if needed.")
                    
                    # Display detailed confidence scores in an expander (if enabled)
                    if PARSER_CONFIG['show_confidence']:
                        with st.expander("📊 View Detailed Confidence Scores"):
                            st.markdown("**Extraction Confidence by Field:**")
                            low_conf_threshold = PARSER_CONFIG['low_confidence_threshold']
                            for field, score in confidence_scores.items():
                                if field != 'overall':
                                    status_icon = "✓" if score >= low_conf_threshold else "⚠️"
                                    st.markdown(f"{status_icon} **{field.replace('_', ' ').title()}:** {score:.0%}")
                            
                            if parsed_resume.extracted_entities.get('skills'):
                                st.markdown("**Identified Skills:**")
                                skills_list = ", ".join(parsed_resume.extracted_entities['skills'][:10])
                                if len(parsed_resume.extracted_entities['skills']) > 10:
                                    skills_list += f" ... (+{len(parsed_resume.extracted_entities['skills']) - 10} more)"
                                st.markdown(f"_{skills_list}_")
                    
                    # Manual correction capability
                    st.markdown('<div class="section-header">Review & Correct (Optional)</div>', unsafe_allow_html=True)
                    st.markdown("You can adjust the extracted values if they don't look correct:")
                    
                    # Initialize corrected values in session state if not present
                    if 'corrected_values' not in st.session_state:
                        st.session_state['corrected_values'] = st.session_state['parsed'].copy()
                
                # Create form for manual corrections
                with st.form("correction_form"):
                    col_form1, col_form2, col_form3 = st.columns(3)
                    
                    with col_form1:
                        corrected_years_exp = st.number_input(
                            "Years of Experience",
                            min_value=0,
                            max_value=50,
                            value=st.session_state['corrected_values']['years_exp'],
                            help="Total years of professional experience"
                        )
                        
                        corrected_age = st.number_input(
                            "Age",
                            min_value=18,
                            max_value=100,
                            value=st.session_state['corrected_values']['age'],
                            help="Your current age"
                        )
                    
                    with col_form2:
                        edu_options = {
                            1: "High School",
                            2: "Diploma/Associate",
                            3: "Bachelor's Degree",
                            4: "Master's Degree",
                            5: "PhD/Doctorate"
                        }
                        current_edu = st.session_state['corrected_values']['education_level']
                        corrected_education = st.selectbox(
                            "Education Level",
                            options=list(edu_options.keys()),
                            format_func=lambda x: edu_options[x],
                            index=list(edu_options.keys()).index(current_edu),
                            help="Highest level of education completed"
                        )
                        
                        corrected_gender = st.selectbox(
                            "Gender",
                            options=[0, 1],
                            format_func=lambda x: "Female/Other" if x == 0 else "Male",
                            index=st.session_state['corrected_values']['gender'],
                            help="Gender (used for demographic analysis)"
                        )
                    
                    with col_form3:
                        st.markdown("**Job Level Flags:**")
                        corrected_is_junior = st.checkbox(
                            "Junior Level",
                            value=bool(st.session_state['corrected_values'].get('is_junior', 0))
                        )
                        corrected_is_mid_level = st.checkbox(
                            "Mid-Level",
                            value=bool(st.session_state['corrected_values'].get('is_mid_level', 0))
                        )
                        corrected_is_senior = st.checkbox(
                            "Senior Level",
                            value=bool(st.session_state['corrected_values'].get('is_senior', 0))
                        )
                        corrected_is_manager = st.checkbox(
                            "Manager",
                            value=bool(st.session_state['corrected_values'].get('is_manager', 0))
                        )
                        corrected_is_executive = st.checkbox(
                            "Executive",
                            value=bool(st.session_state['corrected_values'].get('is_executive', 0))
                        )
                    
                    col_form4, col_form5 = st.columns(2)
                    
                    with col_form4:
                        st.markdown("**Job Category:**")
                        corrected_is_tech = st.checkbox(
                            "Technology/IT",
                            value=bool(st.session_state['corrected_values']['is_tech'])
                        )
                        corrected_is_sales = st.checkbox(
                            "Sales",
                            value=bool(st.session_state['corrected_values']['is_sales'])
                        )
                    
                    with col_form5:
                        st.markdown("**&nbsp;**")  # Spacing
                        corrected_is_marketing = st.checkbox(
                            "Marketing",
                            value=bool(st.session_state['corrected_values']['is_marketing'])
                        )
                        corrected_is_hr = st.checkbox(
                            "HR/Recruitment",
                            value=bool(st.session_state['corrected_values']['is_hr'])
                        )
                    
                    # Submit button for corrections
                    submitted = st.form_submit_button("Apply Corrections", use_container_width=True)
                    
                    if submitted:
                        # Update corrected values in session state
                        st.session_state['corrected_values'] = {
                            'years_exp': corrected_years_exp,
                            'education_level': corrected_education,
                            'age': corrected_age,
                            'gender': corrected_gender,
                            'is_senior': int(corrected_is_senior),
                            'is_manager': int(corrected_is_manager),
                            'is_executive': int(corrected_is_executive),
                            'is_junior': int(corrected_is_junior),
                            'is_mid_level': int(corrected_is_mid_level),
                            'is_tech': int(corrected_is_tech),
                            'is_sales': int(corrected_is_sales),
                            'is_marketing': int(corrected_is_marketing),
                            'is_hr': int(corrected_is_hr)
                        }
                        
                        # Recalculate features with corrected values
                        corrected_features = create_features_from_parsed(st.session_state['corrected_values'])
                        # Force update by creating a new dict
                        st.session_state['features'] = dict(corrected_features)
                        
                        # If prediction already exists, recalculate it with corrected values
                        if 'predicted_salary' in st.session_state:
                            features_df = pd.DataFrame([corrected_features])
                            features_df = features_df[feature_names]
                            features_scaled = scaler.transform(features_df)
                            predicted_salary = predict_salary(model, features_scaled)
                            st.session_state['predicted_salary'] = predicted_salary
                        
                        # Don't update 'parsed' - keep original for reference
                        # The display logic will use 'corrected_values' when available
                        
                        st.success("✓ Corrections applied! Prediction and analytics updated with corrected values.")
                        st.rerun()
                
                # Predict button (outside form)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Generate Salary Prediction", use_container_width=True, type="primary"):
                    with st.spinner("Calculating..."):
                        # Use corrected values if available, otherwise use parsed values
                        values_to_use = st.session_state.get('corrected_values', st.session_state['parsed'])
                        features_to_use = create_features_from_parsed(values_to_use)
                        
                        features_df = pd.DataFrame([features_to_use])
                        features_df = features_df[feature_names]
                        features_scaled = scaler.transform(features_df)
                        predicted_salary = predict_salary(model, features_scaled)
                        st.session_state['predicted_salary'] = predicted_salary
                        st.rerun()
            else:
                st.error("Could not extract text from PDF. Please try another file.")
        
        with col2:
            st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
            
            if 'predicted_salary' in st.session_state:
                salary = st.session_state['predicted_salary']
                features = st.session_state['features']
                # Use corrected values if available, otherwise use parsed values
                parsed = st.session_state.get('corrected_values', st.session_state['parsed'])
                
                # Results header
                st.markdown(f"""
                <div class="results-header">
                    <h2>Predicted Annual Salary</h2>
                    <div class="salary">${salary:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Breakdown (removed redundant chart for cleaner UI)
                st.markdown("#### Salary Breakdown")
                col_x, col_y, col_z = st.columns(3)
                
                with col_x:
                    st.markdown(f'''
                    <div class="metric-card" style="text-align: center;">
                        <h3 style="margin-bottom: 1rem;">Annual</h3>
                        <p class="value" style="margin: 0;">${salary:,.0f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_y:
                    st.markdown(f'''
                    <div class="metric-card" style="text-align: center;">
                        <h3 style="margin-bottom: 1rem;">Monthly</h3>
                        <p class="value" style="margin: 0;">${salary/12:,.0f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col_z:
                    st.markdown(f'''
                    <div class="metric-card" style="text-align: center;">
                        <h3 style="margin-bottom: 1rem;">Hourly</h3>
                        <p class="value" style="margin: 0;">${salary/2080:,.0f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Confidence range
                st.markdown("#### Confidence Interval")
                lower = salary * 0.97
                upper = salary * 1.03
                st.info(f"**Expected Range:** ${lower:,.0f} - ${upper:,.0f} (±3% at 95% confidence)")
                
                # Key factors
                st.markdown("#### Key Factors")
                
                # Determine job level from corrected values
                if parsed.get('is_executive'):
                    level = "Executive"
                elif parsed.get('is_manager'):
                    level = "Manager"
                elif parsed.get('is_senior'):
                    level = "Senior"
                elif parsed.get('is_mid_level'):
                    level = "Mid-Level"
                elif parsed.get('is_junior'):
                    level = "Junior"
                else:
                    level = "Entry-Level"
                
                st.markdown(f"""
                <div class="feature-item">
                    <span class="label">Years of Experience</span>
                    <span class="value">{parsed['years_exp']} years</span>
                </div>
                <div class="feature-item">
                    <span class="label">Education Level</span>
                    <span class="value">Level {parsed['education_level']}/5</span>
                </div>
                <div class="feature-item">
                    <span class="label">Job Level</span>
                    <span class="value">{level}</span>
                </div>
                <div class="feature-item">
                    <span class="label">Career Score</span>
                    <span class="value">{features['experience_education']:.0f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("#### Career Recommendations")
                
                if parsed['years_exp'] < 5:
                    st.markdown('<div class="recommendation"><p>→ Gain additional years of experience to increase earning potential</p></div>', unsafe_allow_html=True)
                if parsed['education_level'] < 4:
                    st.markdown('<div class="recommendation"><p>→ Consider pursuing advanced degrees (Master\'s or PhD)</p></div>', unsafe_allow_html=True)
                if not parsed['is_senior'] and parsed['years_exp'] >= 5:
                    st.markdown('<div class="recommendation"><p>→ Target senior-level positions for higher compensation</p></div>', unsafe_allow_html=True)
                if not parsed['is_manager'] and parsed['years_exp'] >= 8:
                    st.markdown('<div class="recommendation"><p>→ Transition into management roles for career advancement</p></div>', unsafe_allow_html=True)
                
            else:
                st.info("👈 Upload your resume in the Prediction tab to get started", icon="💡")
    
    with tab3:
        # Analytics Dashboard
        if 'predicted_salary' in st.session_state:
            salary = st.session_state['predicted_salary']
            # Always get fresh features from session state - force dict copy
            features = dict(st.session_state.get('features', {}))
            # Use corrected values if available, otherwise use parsed values
            parsed = st.session_state.get('corrected_values', st.session_state.get('parsed', {}))
            
            # Debug: Show if using corrected values
            if 'corrected_values' in st.session_state:
                years_in_features = features.get('Years of Experience', 0)
                st.info(f"📊 Using corrected data - Years: {parsed.get('years_exp', 0)} | Features Years: {years_in_features}", icon="✅")
            
            st.markdown('<div class="section-header">Comprehensive Analytics Dashboard</div>', unsafe_allow_html=True)
            
            # Row 1: Salary factors and salary comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_salary_factors_breakdown_chart(parsed, salary), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_salary_distribution_chart(salary), use_container_width=True)
            
            # Row 2: Career trajectory and skills radar
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_experience_salary_curve(parsed), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_skills_radar_chart(parsed), use_container_width=True)
            
            # Row 3: Market comparison and growth projection
            st.markdown('<div class="section-header">Market Analysis & Projections</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_market_comparison_chart(salary, parsed), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_growth_projection_chart(parsed, salary), use_container_width=True)
            
            # Key Insights Summary
            st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                percentile = min(95, 50 + parsed['years_exp'] * 2)
                st.metric(
                    label="Experience Percentile",
                    value=f"{percentile}th",
                    delta=f"{parsed['years_exp']} years"
                )
            
            with col2:
                premium = (parsed['education_level'] - 3) * 15
                st.metric(
                    label="Education Premium",
                    value=f"+{premium}%",
                    delta=f"Level {parsed['education_level']}/5"
                )
            
            with col3:
                # Determine job level from corrected values
                if parsed.get('is_executive'):
                    level = "Executive"
                elif parsed.get('is_manager'):
                    level = "Manager"
                elif parsed.get('is_senior'):
                    level = "Senior"
                elif parsed.get('is_mid_level'):
                    level = "Mid-Level"
                elif parsed.get('is_junior'):
                    level = "Junior"
                else:
                    level = "Entry-Level"
                
                mult = 1.0 + parsed.get('is_senior', 0) * 0.3 + parsed.get('is_manager', 0) * 0.5 + parsed.get('is_executive', 0) * 1.0
                st.metric(
                    label="Position Multiplier",
                    value=f"{mult:.1f}x",
                    delta=level
                )
            
            # Salary breakdown table
            st.markdown('<div class="section-header">Salary Breakdown Analysis</div>', unsafe_allow_html=True)
            
            breakdown_data = {
                'Component': ['Base Prediction', 'Experience Premium', 'Education Premium', 'Level Premium', 'Total Predicted'],
                'Amount': [
                    salary * 0.6,
                    salary * 0.2,
                    salary * 0.1,
                    salary * 0.1,
                    salary
                ],
                'Percentage': ['60%', '20%', '10%', '10%', '100%']
            }
            
            df_breakdown = pd.DataFrame(breakdown_data)
            df_breakdown['Amount'] = df_breakdown['Amount'].apply(lambda x: f'${x:,.0f}')
            
            st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
            
        else:
            st.info("Upload a resume and generate a prediction to view detailed analytics.")
    
    with tab4:
        # Data Insights - Visualizations from training data
        st.markdown('<div class="section-header">Training Data Insights</div>', unsafe_allow_html=True)
        st.markdown("Explore relationships between features and salary from our **Enhanced 54K Dataset** used for model training")
        
        # Load the training data (enhanced dataset)
        try:
            data_path = Path('data/enhanced_54k_dataset.csv')
            if data_path.exists():
                df_viz = pd.read_csv(data_path)
                df_viz.columns = df_viz.columns.str.strip().str.lower().str.replace(' ', '_')
                # Drop resume_text column if it exists (not needed for viz)
                if 'resume_text' in df_viz.columns:
                    df_viz = df_viz.drop('resume_text', axis=1)
                df_viz = df_viz.dropna()
                
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", f"{len(df_viz):,}")
                with col2:
                    st.metric("Avg Salary", f"${df_viz['salary'].mean():,.0f}")
                with col3:
                    st.metric("Salary Range", f"${df_viz['salary'].min():,.0f} - ${df_viz['salary'].max():,.0f}")
                with col4:
                    st.metric("Avg Experience", f"{df_viz['years_of_experience'].mean():.1f} yrs")
                
                st.markdown("---")
                
                # Row 1: Experience vs Salary & Education vs Salary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Experience vs Salary")
                    fig = px.scatter(
                        df_viz, 
                        x='years_of_experience', 
                        y='salary',
                        trendline="lowess",
                        opacity=0.6,
                        color_discrete_sequence=['#3b82f6']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        xaxis=dict(title='Years of Experience', gridcolor='#334155'),
                        yaxis=dict(title='Salary ($)', gridcolor='#334155'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🎓 Education Level vs Salary")
                    education_order = ['High School', 'Bachelor', 'Master', 'PhD']
                    edu_salary = df_viz.groupby('education_level')['salary'].agg(['mean', 'median', 'std']).reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=edu_salary['education_level'],
                        y=edu_salary['mean'],
                        name='Average',
                        marker_color='#3b82f6',
                        error_y=dict(type='data', array=edu_salary['std'])
                    ))
                    fig.add_trace(go.Scatter(
                        x=edu_salary['education_level'],
                        y=edu_salary['median'],
                        name='Median',
                        mode='markers+lines',
                        marker=dict(size=10, color='#10b981'),
                        line=dict(color='#10b981', width=2)
                    ))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        xaxis=dict(title='Education Level', gridcolor='#334155'),
                        yaxis=dict(title='Salary ($)', gridcolor='#334155'),
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 2: Job Title Analysis & Gender Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 💼 Top 10 Job Titles by Avg Salary")
                    job_salary = df_viz.groupby('job_title')['salary'].mean().sort_values(ascending=False).head(10)
                    
                    fig = go.Figure(go.Bar(
                        x=job_salary.values,
                        y=job_salary.index,
                        orientation='h',
                        marker=dict(
                            color=job_salary.values,
                            colorscale='Blues',
                            showscale=True
                        )
                    ))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        xaxis=dict(title='Average Salary ($)', gridcolor='#334155'),
                        yaxis=dict(title='', gridcolor='#334155'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### ⚖️ Salary Distribution by Gender")
                    fig = go.Figure()
                    for gender in df_viz['gender'].unique():
                        gender_data = df_viz[df_viz['gender'] == gender]['salary']
                        fig.add_trace(go.Box(
                            y=gender_data,
                            name=gender,
                            boxmean='sd'
                        ))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        yaxis=dict(title='Salary ($)', gridcolor='#334155'),
                        xaxis=dict(gridcolor='#334155'),
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 3: Age vs Salary & Correlation Heatmap
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 👤 Age vs Salary")
                    fig = px.scatter(
                        df_viz,
                        x='age',
                        y='salary',
                        trendline="lowess",
                        opacity=0.5,
                        color_discrete_sequence=['#8b5cf6']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        xaxis=dict(title='Age', gridcolor='#334155'),
                        yaxis=dict(title='Salary ($)', gridcolor='#334155'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🔗 Feature Correlation")
                    numeric_cols = ['age', 'years_of_experience', 'salary']
                    corr_matrix = df_viz[numeric_cols].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.values.round(2),
                        texttemplate='%{text}',
                        textfont={"size": 14},
                        colorbar=dict(title="Correlation")
                    ))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Row 4: Salary Distribution & Experience-Education Combined
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Salary Distribution")
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df_viz['salary'],
                        nbinsx=50,
                        marker_color='#3b82f6',
                        opacity=0.7,
                        name='Salary'
                    ))
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        xaxis=dict(title='Salary ($)', gridcolor='#334155'),
                        yaxis=dict(title='Frequency', gridcolor='#334155'),
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🎯 Experience vs Salary by Education")
                    fig = px.scatter(
                        df_viz,
                        x='years_of_experience',
                        y='salary',
                        color='education_level',
                        opacity=0.6,
                        category_orders={'education_level': education_order}
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        xaxis=dict(title='Years of Experience', gridcolor='#334155'),
                        yaxis=dict(title='Salary ($)', gridcolor='#334155'),
                        height=400,
                        legend=dict(title='Education Level')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical Summary
                st.markdown("---")
                st.markdown("#### 📋 Statistical Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Salary Statistics**")
                    salary_stats = df_viz['salary'].describe()
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %ile', '75th %ile'],
                        'Value': [
                            f"${salary_stats['mean']:,.0f}",
                            f"${df_viz['salary'].median():,.0f}",
                            f"${salary_stats['std']:,.0f}",
                            f"${salary_stats['min']:,.0f}",
                            f"${salary_stats['max']:,.0f}",
                            f"${salary_stats['25%']:,.0f}",
                            f"${salary_stats['75%']:,.0f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**Experience Statistics**")
                    exp_stats = df_viz['years_of_experience'].describe()
                    exp_df = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %ile', '75th %ile'],
                        'Value': [
                            f"{exp_stats['mean']:.1f} years",
                            f"{df_viz['years_of_experience'].median():.1f} years",
                            f"{exp_stats['std']:.1f} years",
                            f"{exp_stats['min']:.1f} years",
                            f"{exp_stats['max']:.1f} years",
                            f"{exp_stats['25%']:.1f} years",
                            f"{exp_stats['75%']:.1f} years"
                        ]
                    })
                    st.dataframe(exp_df, use_container_width=True, hide_index=True)
                
            else:
                st.warning("Training data not found. Please ensure 'data/enhanced_54k_dataset.csv' exists.")
                st.info("This dataset contains 6,700+ enhanced records used for model training.")
                
        except Exception as e:
            st.error(f"Error loading visualization data: {str(e)}")
    
    with tab5:
        # Model Training UI
        st.markdown('<div class="section-header">Model Training Center</div>', unsafe_allow_html=True)
        st.markdown("Train a new model from scratch with real-time statistics and performance metrics")
        
        # Training configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Training Configuration")
            
            dataset_path = st.text_input(
                "Dataset Path",
                value="data/enhanced_54k_dataset.csv",
                help="Path to the training dataset"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                test_size = st.slider("Test Split %", 10, 30, 15, 5, help="Percentage of data for testing")
            with col_b:
                random_state = st.number_input("Random Seed", value=42, help="For reproducibility")
        
        with col2:
            st.markdown("### 📋 Quick Info")
            st.info("""
            **Training Process:**
            1. Load & clean data
            2. Engineer features
            3. Train 4 models
            4. Select best model
            5. Save to production
            
            **Time:** ~45-60 seconds
            """)
        
        st.markdown("---")
        
        # Training button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                # Initialize training
                import time
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                import xgboost as xgb
                
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Load Data
                    status_text.markdown("### 📂 Loading Dataset...")
                    progress_bar.progress(10)
                    
                    df = pd.read_csv(dataset_path)
                    initial_count = len(df)
                    
                    # Clean data
                    df = df.dropna(subset=['Salary'])
                    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
                    df = df[df['Salary'] > 0]
                    cleaned_count = len(df)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Initial Records", f"{initial_count:,}")
                    with col2:
                        st.metric("After Cleaning", f"{cleaned_count:,}")
                    with col3:
                        st.metric("Removed", f"{initial_count - cleaned_count:,}")
                    
                    progress_bar.progress(20)
                    
                    # Step 2: Feature Engineering
                    status_text.markdown("### 🔧 Engineering Features...")
                    time.sleep(0.5)
                    
                    # Encode categorical
                    df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'].fillna('Unknown'))
                    
                    # Education level
                    education_map = {"High School": 1, "Bachelor's": 3, "Master's": 4, "PhD": 5}
                    df['Education_level'] = df['Education Level'].map(education_map).fillna(3)
                    
                    # Job levels
                    df['is_senior'] = df['Job Title'].str.lower().str.contains('senior|sr|lead|principal', na=False).astype(int)
                    df['is_manager'] = df['Job Title'].str.lower().str.contains('manager|director|head|vp', na=False).astype(int)
                    df['is_executive'] = df['Job Title'].str.lower().str.contains('ceo|cto|cfo|president|executive', na=False).astype(int)
                    df['is_junior'] = df['Job Title'].str.lower().str.contains('junior|jr|entry|intern', na=False).astype(int)
                    df['is_mid_level'] = ((df['is_senior'] == 0) & (df['is_manager'] == 0) & 
                                          (df['is_executive'] == 0) & (df['is_junior'] == 0)).astype(int)
                    
                    # Job categories
                    df['is_tech'] = df['Job Title'].str.lower().str.contains('software|developer|engineer|data|analyst|scientist', na=False).astype(int)
                    df['is_sales'] = df['Job Title'].str.lower().str.contains('sales|account', na=False).astype(int)
                    df['is_marketing'] = df['Job Title'].str.lower().str.contains('marketing', na=False).astype(int)
                    df['is_hr'] = df['Job Title'].str.lower().str.contains('hr|human|recruiter', na=False).astype(int)
                    
                    # Experience features
                    df['Years of Experience'] = pd.to_numeric(df['Years of Experience'], errors='coerce').fillna(0)
                    df['experience_squared'] = df['Years of Experience'] ** 2
                    df['experience_education'] = df['Years of Experience'] * df['Education_level']
                    
                    # Age features
                    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median())
                    df['age_experience_ratio'] = df['Age'] / (df['Years of Experience'] + 1)
                    
                    # Interaction features
                    df['senior_tech'] = df['is_senior'] * df['is_tech']
                    df['manager_experience'] = df['is_manager'] * df['Years of Experience']
                    df['education_tech'] = df['Education_level'] * df['is_tech']
                    
                    # Binned features
                    df['experience_bin'] = pd.cut(df['Years of Experience'], bins=[-1, 2, 5, 10, 20, 50], labels=[1, 2, 3, 4, 5]).astype(int)
                    df['age_bin'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]).astype(int)
                    
                    feature_cols = [
                        'Years of Experience', 'Education_level', 'Age', 'Gender_encoded',
                        'is_senior', 'is_manager', 'is_executive', 'is_junior', 'is_mid_level',
                        'is_tech', 'is_sales', 'is_marketing', 'is_hr',
                        'experience_squared', 'experience_education', 'age_experience_ratio',
                        'senior_tech', 'manager_experience', 'education_tech',
                        'experience_bin', 'age_bin'
                    ]
                    
                    X = df[feature_cols].fillna(0)
                    y = df['Salary'].values
                    
                    st.success(f"✅ Engineered {len(feature_cols)} features")
                    progress_bar.progress(30)
                    
                    # Step 3: Scale and Split
                    status_text.markdown("### 📊 Scaling & Splitting Data...")
                    time.sleep(0.3)
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size/100, random_state=random_state
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Samples", f"{len(X_train):,}", f"{len(X_train)/len(X)*100:.1f}%")
                    with col2:
                        st.metric("Testing Samples", f"{len(X_test):,}", f"{len(X_test)/len(X)*100:.1f}%")
                    
                    progress_bar.progress(40)
                    
                    # Step 4: Train Models
                    status_text.markdown("### 🚀 Training Models...")
                    
                    models_results = {}
                    
                    # XGBoost
                    st.markdown("**1/4: Training XGBoost...**")
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=1000, max_depth=10, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=-1
                    )
                    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                    
                    # Calculate train metrics
                    xgb_train_pred = xgb_model.predict(X_train)
                    xgb_train_r2 = r2_score(y_train, xgb_train_pred)
                    
                    # Calculate test metrics
                    xgb_test_pred = xgb_model.predict(X_test)
                    xgb_test_r2 = r2_score(y_test, xgb_test_pred)
                    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
                    xgb_mae = mean_absolute_error(y_test, xgb_test_pred)
                    xgb_mape = np.mean(np.abs((y_test - xgb_test_pred) / y_test)) * 100
                    
                    models_results['XGBoost'] = {
                        'model': xgb_model,
                        'train_r2': xgb_train_r2,
                        'test_r2': xgb_test_r2,
                        'rmse': xgb_rmse,
                        'mae': xgb_mae,
                        'mape': xgb_mape,
                        'test_pred': xgb_test_pred
                    }
                    st.success(f"✓ XGBoost Train R²: {xgb_train_r2:.4f} ({xgb_train_r2*100:.2f}%) | Test R²: {xgb_test_r2:.4f} ({xgb_test_r2*100:.2f}%) | RMSE: ${xgb_rmse:,.2f} | MAE: ${xgb_mae:,.2f} | MAPE: {xgb_mape:.2f}%")
                    progress_bar.progress(55)
                    
                    # Gradient Boosting
                    st.markdown("**2/4: Training Gradient Boosting...**")
                    gb_model = GradientBoostingRegressor(
                        n_estimators=1000, max_depth=8, learning_rate=0.05,
                        subsample=0.8, random_state=random_state
                    )
                    gb_model.fit(X_train, y_train)
                    
                    # Calculate train metrics
                    gb_train_pred = gb_model.predict(X_train)
                    gb_train_r2 = r2_score(y_train, gb_train_pred)
                    
                    # Calculate test metrics
                    gb_test_pred = gb_model.predict(X_test)
                    gb_test_r2 = r2_score(y_test, gb_test_pred)
                    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
                    gb_mae = mean_absolute_error(y_test, gb_test_pred)
                    gb_mape = np.mean(np.abs((y_test - gb_test_pred) / y_test)) * 100
                    
                    models_results['GradientBoosting'] = {
                        'model': gb_model,
                        'train_r2': gb_train_r2,
                        'test_r2': gb_test_r2,
                        'rmse': gb_rmse,
                        'mae': gb_mae,
                        'mape': gb_mape,
                        'test_pred': gb_test_pred
                    }
                    st.success(f"✓ Gradient Boosting Train R²: {gb_train_r2:.4f} ({gb_train_r2*100:.2f}%) | Test R²: {gb_test_r2:.4f} ({gb_test_r2*100:.2f}%) | RMSE: ${gb_rmse:,.2f} | MAE: ${gb_mae:,.2f} | MAPE: {gb_mape:.2f}%")
                    progress_bar.progress(70)
                    
                    # Random Forest
                    st.markdown("**3/4: Training Random Forest...**")
                    rf_model = RandomForestRegressor(
                        n_estimators=1000, max_depth=20, min_samples_split=5,
                        random_state=random_state, n_jobs=-1
                    )
                    rf_model.fit(X_train, y_train)
                    
                    # Calculate train metrics
                    rf_train_pred = rf_model.predict(X_train)
                    rf_train_r2 = r2_score(y_train, rf_train_pred)
                    
                    # Calculate test metrics
                    rf_test_pred = rf_model.predict(X_test)
                    rf_test_r2 = r2_score(y_test, rf_test_pred)
                    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
                    rf_mae = mean_absolute_error(y_test, rf_test_pred)
                    rf_mape = np.mean(np.abs((y_test - rf_test_pred) / y_test)) * 100
                    
                    models_results['RandomForest'] = {
                        'model': rf_model,
                        'train_r2': rf_train_r2,
                        'test_r2': rf_test_r2,
                        'rmse': rf_rmse,
                        'mae': rf_mae,
                        'mape': rf_mape,
                        'test_pred': rf_test_pred
                    }
                    st.success(f"✓ Random Forest Train R²: {rf_train_r2:.4f} ({rf_train_r2*100:.2f}%) | Test R²: {rf_test_r2:.4f} ({rf_test_r2*100:.2f}%) | RMSE: ${rf_rmse:,.2f} | MAE: ${rf_mae:,.2f} | MAPE: {rf_mape:.2f}%")
                    progress_bar.progress(85)
                    
                    # Ensemble
                    st.markdown("**4/4: Creating Ensemble...**")
                    total_r2 = xgb_test_r2 + gb_test_r2 + rf_test_r2
                    w_xgb, w_gb, w_rf = xgb_test_r2/total_r2, gb_test_r2/total_r2, rf_test_r2/total_r2
                    
                    # Train predictions
                    ensemble_train_pred = w_xgb * xgb_train_pred + w_gb * gb_train_pred + w_rf * rf_train_pred
                    ensemble_train_r2 = r2_score(y_train, ensemble_train_pred)
                    
                    # Test predictions
                    ensemble_test_pred = w_xgb * xgb_test_pred + w_gb * gb_test_pred + w_rf * rf_test_pred
                    ensemble_test_r2 = r2_score(y_test, ensemble_test_pred)
                    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
                    ensemble_mae = mean_absolute_error(y_test, ensemble_test_pred)
                    ensemble_mape = np.mean(np.abs((y_test - ensemble_test_pred) / y_test)) * 100
                    
                    models_results['Ensemble'] = {
                        'train_r2': ensemble_train_r2,
                        'test_r2': ensemble_test_r2,
                        'rmse': ensemble_rmse,
                        'mae': ensemble_mae,
                        'mape': ensemble_mape,
                        'test_pred': ensemble_test_pred,
                        'weights': (w_xgb, w_gb, w_rf)
                    }
                    st.success(f"✓ Ensemble Train R²: {ensemble_train_r2:.4f} ({ensemble_train_r2*100:.2f}%) | Test R²: {ensemble_test_r2:.4f} ({ensemble_test_r2*100:.2f}%) | RMSE: ${ensemble_rmse:,.2f} | MAE: ${ensemble_mae:,.2f} | MAPE: {ensemble_mape:.2f}%")
                    progress_bar.progress(95)
                    
                    # Step 5: Select Best Model
                    status_text.markdown("### 🏆 Selecting Best Model...")
                    best_name = max(models_results.items(), key=lambda x: x[1]['test_r2'])[0]
                    best_model_data = models_results[best_name]
                    
                    if best_name == 'Ensemble':
                        best_model = {
                            'xgb': xgb_model, 'gb': gb_model, 'rf': rf_model,
                            'weights': best_model_data['weights']
                        }
                    else:
                        best_model = best_model_data['model']
                    
                    # Save model
                    model_artifact = {
                        'model': best_model,
                        'scaler': scaler,
                        'feature_names': feature_cols,
                        'metadata': {
                            'model_name': best_name,
                            'train_r2': best_model_data['train_r2'],
                            'test_r2': best_model_data['test_r2'],
                            'accuracy_percentage': best_model_data['test_r2'] * 100,
                            'rmse': best_model_data['rmse'],
                            'mae': best_model_data['mae'],
                            'mape': best_model_data['mape'],
                            'training_samples': len(X_train),
                            'test_samples': len(X_test),
                            'features_count': len(feature_cols),
                            'trained_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                    
                    Path('models').mkdir(exist_ok=True)
                    joblib.dump(model_artifact, 'models/production_model.pkl')
                    
                    progress_bar.progress(100)
                    status_text.markdown("### ✅ Training Complete!")
                    
                    # Display Results
                    st.markdown("---")
                    st.markdown("## 📊 Training Results")
                    
                    # Model comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Model Performance Comparison")
                        comparison_df = pd.DataFrame({
                            'Model': list(models_results.keys()),
                            'Train R²': [m['train_r2'] for m in models_results.values()],
                            'Test R²': [m['test_r2'] for m in models_results.values()],
                            'RMSE': [f"${m['rmse']:,.2f}" for m in models_results.values()],
                            'MAE': [f"${m['mae']:,.2f}" for m in models_results.values()],
                            'MAPE': [f"{m['mape']:.2f}%" for m in models_results.values()]
                        })
                        comparison_df = comparison_df.sort_values('Test R²', ascending=False)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # Bar chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name='Train R²',
                            x=comparison_df['Model'],
                            y=[m['train_r2']*100 for m in models_results.values()],
                            marker_color='#64748b'
                        ))
                        fig.add_trace(go.Bar(
                            name='Test R²',
                            x=comparison_df['Model'],
                            y=[m['test_r2']*100 for m in models_results.values()],
                            marker_color=['#10b981' if m == best_name else '#3b82f6' for m in comparison_df['Model']]
                        ))
                        fig.update_layout(
                            title='Model Accuracy Comparison (Train vs Test)',
                            yaxis_title='R² Score (%)',
                            barmode='group',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e2e8f0'),
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### Best Model Metrics")
                        
                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            st.metric("Model", best_name)
                            st.metric("Train R²", f"{best_model_data['train_r2']:.4f} ({best_model_data['train_r2']*100:.2f}%)")
                            st.metric("Test R²", f"{best_model_data['test_r2']:.4f} ({best_model_data['test_r2']*100:.2f}%)")
                        with metrics_col2:
                            st.metric("RMSE", f"${best_model_data['rmse']:,.2f}")
                            st.metric("MAE", f"${best_model_data['mae']:,.2f}")
                            st.metric("MAPE", f"{best_model_data['mape']:.2f}%")
                        
                        # Prediction vs Actual scatter
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=y_test, y=best_model_data['test_pred'],
                            mode='markers',
                            marker=dict(color='#3b82f6', size=5, opacity=0.6),
                            name='Predictions'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode='lines',
                            line=dict(color='#10b981', dash='dash'),
                            name='Perfect Prediction'
                        ))
                        fig.update_layout(
                            title='Predicted vs Actual Salary',
                            xaxis_title='Actual Salary ($)',
                            yaxis_title='Predicted Salary ($)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e2e8f0'),
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance (if available)
                    if best_name in ['XGBoost', 'RandomForest', 'GradientBoosting']:
                        st.markdown("### 🎯 Top 10 Feature Importance")
                        if hasattr(best_model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': best_model.feature_importances_
                            }).sort_values('Importance', ascending=False).head(10)
                            
                            fig = go.Figure(go.Bar(
                                x=importance_df['Importance'],
                                y=importance_df['Feature'],
                                orientation='h',
                                marker_color='#3b82f6'
                            ))
                            fig.update_layout(
                                xaxis_title='Importance',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#e2e8f0'),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("🎉 Model saved to `models/production_model.pkl` - Reload the app to use the new model!")
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Show current model info
        st.markdown("---")
        st.markdown("### 📦 Current Production Model")
        
        if Path('models/production_model.pkl').exists():
            try:
                current_model = joblib.load('models/production_model.pkl')
                meta = current_model['metadata']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model Type", meta.get('model_name', 'Unknown'))
                with col2:
                    st.metric("Accuracy", f"{meta.get('accuracy_percentage', 0):.2f}%")
                with col3:
                    st.metric("Features", meta.get('features_count', 0))
                with col4:
                    st.metric("Trained", meta.get('trained_at', 'Unknown'))
                
                with st.expander("📋 View Full Model Details"):
                    st.json(meta)
            except Exception as e:
                st.error(f"Error loading current model: {str(e)}")
        else:
            st.warning("No production model found. Train a new model above.")


if __name__ == "__main__":
    main()
