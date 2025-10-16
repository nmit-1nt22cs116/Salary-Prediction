import streamlit as st
import joblib  # Changed from pickle to joblib
import tempfile
import os
from datetime import datetime
import PyPDF2
import numpy as np
import spacy
from spacy.matcher import Matcher
import re

# ----------------- NLP Setup -----------------
nlp = spacy.load("en_core_web_sm")

# Predefined skill database
skill_database = [
    "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Rust", "PHP", "Swift",
    "Kotlin", "R", "Scala", "TypeScript", "React", "Angular", "Vue", "Node.js",
    "Django", "Flask", "FastAPI", "Spring", "Express", "Next.js", "HTML", "CSS",
    "Bootstrap", "Tailwind", "SQL", "MongoDB", "PostgreSQL", "MySQL", "Redis",
    "Elasticsearch", "Cassandra", "DynamoDB", "Oracle", "SQLite", "TensorFlow",
    "PyTorch", "Scikit-learn", "Keras", "NLP", "Computer Vision", "OpenCV", "Pandas",
    "NumPy", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Jenkins",
    "Terraform", "Ansible", "Git", "Jira", "Linux", "Agile", "Scrum", "REST API",
    "GraphQL", "Microservices", "Kafka", "RabbitMQ", "Tableau", "Power BI", "Excel",
    "Data Analysis", "Data Visualization", "Statistics", "A/B Testing", "Android",
    "iOS", "React Native", "Flutter", "Xamarin"
]

matcher = Matcher(nlp.vocab)
patterns = [[{"LOWER": skill.lower()}] for skill in skill_database]
for i, pattern in enumerate(patterns):
    matcher.add(f"SKILL_{i}", [pattern])


# ----------------- NLP Functions -----------------
def pdf_to_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def extract_contact_info(text):
    contact = {}
    # Email
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if emails: contact['Email'] = emails[0]
    # Phone
    phones = re.findall(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if phones: contact['Phone'] = phones[0]
    # LinkedIn
    linkedin = re.findall(r'(?:linkedin\.com/in/|linkedin\.com/pub/)([A-Za-z0-9\-]+)', text, re.I)
    if linkedin: contact['LinkedIn'] = f"linkedin.com/in/{linkedin[0]}"
    # GitHub
    github = re.findall(r'(?:github\.com/)([A-Za-z0-9\-]+)', text, re.I)
    if github: contact['GitHub'] = f"github.com/{github[0]}"
    return contact


def extract_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    skills_found = set()
    for match_id, start, end in matches:
        skills_found.add(doc[start:end].text)
    return list(skills_found) if skills_found else ["Skills not specified"]


def extract_education(text):
    doc = nlp(text)
    education = []
    degree_patterns = ["B.Tech", "B.E", "Bachelor", "M.Tech", "M.S", "Master", "MBA", "PhD", "Doctorate", "Diploma"]
    for token in doc:
        if any(d.lower() in token.text.lower() for d in degree_patterns):
            phrase = token.text
            education.append(phrase)
    institutions = [ent.text for ent in doc.ents if ent.label_ in ["ORG"]]
    education.extend(institutions)
    return list(set(education)) if education else ["Education details not specified"]


def extract_experience(text):
    doc = nlp(text)
    experiences = []
    job_titles = ["Engineer", "Developer", "Manager", "Analyst", "Consultant", "Designer", "Architect", "Lead",
                  "Director", "Specialist", "Coordinator"]
    for token in doc:
        if any(j.lower() in token.text.lower() for j in job_titles):
            experiences.append(token.text)
    # Extract total years
    total_years = 0
    current_year = datetime.now().year
    date_patterns = [
        r'(\d{4})\s*[-—–]\s*(\d{4}|Present|Current|Till date)',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\s*[-—–]\s*(?:[A-Za-z]+\s+)?(\d{4}|Present|Current)'
    ]
    year_ranges = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.I)
        for start, end in matches:
            start_year = int(start)
            end_year = current_year if end.lower() in ["present", "current", "till date"] else int(end)
            if 1980 <= start_year <= end_year <= current_year + 5:
                year_ranges.append((start_year, end_year))
    if year_ranges:
        year_ranges.sort()
        merged_ranges = [year_ranges[0]]
        for start, end in year_ranges[1:]:
            if start <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(end, merged_ranges[-1][1]))
            else:
                merged_ranges.append((start, end))
        total_years = sum(end - start for start, end in merged_ranges)
    return experiences if experiences else ["Experience not specified"], total_years


def extract_certifications(text):
    cert_keywords = ["AWS Certified", "Microsoft Certified", "Google Cloud", "Oracle Certified", "PMP", "CISSP",
                     "CompTIA", "Certification"]
    certs = [line.strip() for line in text.split("\n") if any(k.lower() in line.lower() for k in cert_keywords)]
    return certs if certs else ["No certifications found"]


def parse_resume_nlp(text):
    parsed = {}
    parsed['contact_info'] = extract_contact_info(text)
    parsed['skills'] = extract_skills(text)
    parsed['education'] = extract_education(text)
    parsed['experience'], parsed['total_experience_years'] = extract_experience(text)
    parsed['certifications'] = extract_certifications(text)
    return parsed


# ----------------- Demo Model -----------------
def create_demo_model():
    from sklearn.linear_model import LinearRegression
    X_train = np.array([
        [2, 10, 0, 1],
        [5, 15, 1, 2],
        [8, 20, 1, 3],
        [3, 12, 0, 2],
        [10, 25, 1, 3],
        [1, 8, 0, 1],
        [7, 18, 1, 2],
        [4, 14, 0, 2],
    ])
    y_train = np.array([30000, 60000, 100000, 45000, 150000, 25000, 85000, 50000])
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Return artifact structure matching train_app.py
    artifact = {
        'model': model,
        'pipeline': {'tfidf': None},  # Demo doesn't use TF-IDF
        'metadata': {
            'model_type': 'LinearRegression (Demo)',
            'is_demo': True
        }
    }
    return artifact


def save_demo_model(path):
    try:
        artifact = create_demo_model()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(artifact, path)  # Use joblib instead of pickle
        return True
    except Exception as e:
        st.error(f"Error saving demo model: {str(e)}")
        return False


def validate_joblib_file(path):
    """Validate if a file is a valid joblib file"""
    try:
        with open(path, 'rb') as f:
            header = f.read(10)
            if not header:
                return False, "File is empty"
        # Try to load it
        joblib.load(path)
        return True, "Valid"
    except Exception as e:
        return False, str(e)


# ----------------- Prediction -----------------
def predict_salary(model_path, resume_text):
    try:
        # Validate file first
        is_valid, message = validate_joblib_file(model_path)
        if not is_valid:
            raise ValueError(f"Invalid model file: {message}")

        # Load artifact using joblib
        artifact = joblib.load(model_path)

        # Extract model and pipeline
        if isinstance(artifact, dict):
            model = artifact.get('model')
            tfidf = artifact.get('pipeline', {}).get('tfidf')
            is_demo = artifact.get('metadata', {}).get('is_demo', False)
        else:
            # Fallback for old format
            model = artifact
            tfidf = None
            is_demo = False

        if model is None:
            raise ValueError("Model not found in artifact")

        # Parse resume
        parsed = parse_resume_nlp(resume_text)

        # Create features based on model type
        if is_demo or tfidf is None:
            # Simple feature-based prediction (demo model)
            features = [
                parsed['total_experience_years'],
                len(parsed['skills']),
                1 if len(parsed['certifications']) > 1 else 0,
                len(parsed['education'])
            ]
            predicted_salary = model.predict([features])[0]
        else:
            # TF-IDF based prediction (trained model)
            resume_tfidf = tfidf.transform([resume_text])
            predicted_salary = model.predict(resume_tfidf)[0]

        return predicted_salary, parsed

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        raise


# ----------------- Streamlit App -----------------
st.set_page_config(page_title="AI Salary Predictor", page_icon="💼", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:2rem;border-radius:10px;color:white;text-align:center;margin-bottom:2rem;">
    <h1>💼 AI Salary Predictor</h1>
    <p>Estimate your salary based on experience, skills, and education using NLP & ML</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    model_option = st.radio("Model Source:", ["Use Demo Model", "Use Default Model Path", "Upload Custom Model"])
    model_path = None

    if model_option == "Use Demo Model":
        demo_path = "demo_model.pkl"
        if not os.path.exists(demo_path):
            if save_demo_model(demo_path):
                st.success("✅ Demo Model Created Successfully")
            else:
                st.error("❌ Failed to create demo model")
        else:
            # Validate existing demo model
            is_valid, msg = validate_joblib_file(demo_path)
            if is_valid:
                st.success("✅ Demo Model Ready")
            else:
                st.warning(f"⚠️ Demo model invalid: {msg}. Recreating...")
                os.remove(demo_path)
                if save_demo_model(demo_path):
                    st.success("✅ Demo Model Recreated Successfully")
        model_path = demo_path

    elif model_option == "Use Default Model Path":
        model_path = st.text_input("Model Path:", "artifacts/salary_model.pkl")
        if model_path:
            if not os.path.exists(model_path):
                st.warning("⚠️ Model file not found at specified path.")
            else:
                is_valid, msg = validate_joblib_file(model_path)
                if is_valid:
                    st.success("✅ Model file found and validated")
                else:
                    st.error(f"❌ Invalid model file: {msg}")
                    model_path = None
    else:
        uploaded_model = st.file_uploader("Upload Model File:", type=["pkl"])
        if uploaded_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                tmp.write(uploaded_model.read())
                model_path = tmp.name

            # Validate uploaded model
            is_valid, msg = validate_joblib_file(model_path)
            if is_valid:
                st.success(f"✅ Model {uploaded_model.name} uploaded and validated")
            else:
                st.error(f"❌ Invalid model file: {msg}")
                model_path = None

with col2:
    input_method = st.radio("Resume Input:", ["Upload PDF", "Paste Text"])
    resume_text = ""
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload PDF:", type=["pdf"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name
            resume_text = pdf_to_text(temp_path)
            if resume_text:
                st.success(f"✅ PDF parsed successfully ({len(resume_text)} characters)")
    else:
        resume_text = st.text_area("Paste Resume Text:", height=300)

if resume_text and model_path and os.path.exists(model_path):
    if st.button("🚀 Predict Salary"):
        with st.spinner("Analyzing Resume..."):
            try:
                salary, parsed = predict_salary(model_path, resume_text)
                st.markdown(
                    f"<h2 style='text-align:center;color:#764ba2;'>💰 Predicted Monthly Salary: ₹{salary:,.2f}</h2>",
                    unsafe_allow_html=True)

                # Metrics
                st.markdown("### Key Metrics")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Experience (Years)", parsed['total_experience_years'])
                metric_col2.metric("Skills Found", len(parsed['skills']))
                metric_col3.metric("Education Entries", len(parsed['education']))
                metric_col4.metric("Certifications",
                                   len([c for c in parsed['certifications'] if c != "No certifications found"]))

                # Tabs
                tab1, tab2 = st.tabs(["📄 Parsed Resume", "💡 Suggestions"])
                with tab1:
                    st.subheader("Extracted Details")
                    st.json(parsed)
                with tab2:
                    st.subheader("Suggestions to Improve Salary")
                    if parsed['total_experience_years'] < 5:
                        st.info("💼 Consider gaining more experience in your field.")
                    if len(parsed['skills']) < 5:
                        st.info("🎯 Add more high-demand skills to your resume.")
                    if len([c for c in parsed['certifications'] if c != "No certifications found"]) == 0:
                        st.info("📜 Consider getting certifications to boost your profile.")
                    if len(parsed['education']) < 2:
                        st.info("🎓 Highlight all educational achievements.")
            except Exception as e:
                st.error(f"Failed to predict salary. Please check your model file.")
else:
    if not resume_text:
        st.info("👆 Please provide resume text or upload a PDF to get started")
    elif not model_path or not os.path.exists(model_path):
        st.info("👈 Please select a valid model to proceed")