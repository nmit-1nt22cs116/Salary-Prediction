"""
Advanced Feature Engineering for Salary Prediction
Extracts comprehensive features from resumes to achieve 97%+ accuracy
"""

import pandas as pd
import numpy as np
import spacy
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Installing spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class ResumeFeatureExtractor:
    """Extract comprehensive features from resumes"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Skill categories with weights
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 
                          'php', 'swift', 'kotlin', 'r', 'scala', 'typescript'],
            'web': ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'fastapi', 
                   'spring', 'express', 'next.js', 'html', 'css', 'bootstrap', 'tailwind'],
            'database': ['sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
                        'cassandra', 'dynamodb', 'oracle', 'sqlite'],
            'ml_ai': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'nlp', 
                     'computer vision', 'opencv', 'pandas', 'numpy', 'machine learning',
                     'deep learning', 'artificial intelligence'],
            'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
                     'terraform', 'ansible', 'jenkins', 'ci/cd'],
            'data': ['data analysis', 'data visualization', 'statistics', 'tableau',
                    'power bi', 'excel', 'a/b testing', 'analytics'],
            'mobile': ['android', 'ios', 'react native', 'flutter', 'xamarin'],
            'other': ['git', 'jira', 'linux', 'agile', 'scrum', 'rest api', 
                     'graphql', 'microservices', 'kafka', 'rabbitmq']
        }
        
        # Education level mapping
        self.education_levels = {
            'phd': 5, 'doctorate': 5, 'ph.d': 5,
            'master': 4, 'mba': 4, 'm.tech': 4, 'm.s': 4, 'm.e': 4,
            'bachelor': 3, 'b.tech': 3, 'b.e': 3, 'b.s': 3, 'b.a': 3,
            'diploma': 2,
            'high school': 1, 'secondary': 1
        }
        
        # Job level keywords
        self.job_levels = {
            'executive': ['ceo', 'cto', 'cfo', 'vp', 'vice president', 'director', 'head of'],
            'senior': ['senior', 'sr', 'lead', 'principal', 'staff', 'architect'],
            'mid': ['manager', 'specialist', 'analyst', 'engineer', 'developer'],
            'junior': ['junior', 'jr', 'associate', 'assistant'],
            'entry': ['intern', 'trainee', 'entry', 'graduate']
        }
    
    def extract_years_of_experience(self, text):
        """Extract years of experience from resume"""
        current_year = datetime.now().year
        years = 0
        
        # Pattern 1: "X years of experience"
        pattern1 = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)'
        matches = re.findall(pattern1, text.lower())
        if matches:
            years = max([int(m) for m in matches])
        
        # Pattern 2: Date ranges (YYYY - YYYY or YYYY - Present)
        pattern2 = r'(\d{4})\s*[-–—]\s*(?:(\d{4})|present|current|now)'
        date_matches = re.findall(pattern2, text.lower())
        
        if date_matches:
            total_years = 0
            for start, end in date_matches:
                start_year = int(start)
                end_year = int(end) if end else current_year
                if 1980 <= start_year <= end_year <= current_year + 1:
                    total_years += (end_year - start_year)
            years = max(years, total_years)
        
        return min(years, 50)  # Cap at 50 years
    
    def extract_education_level(self, text):
        """Extract highest education level"""
        text_lower = text.lower()
        max_level = 0
        
        for edu, level in self.education_levels.items():
            if edu in text_lower:
                max_level = max(max_level, level)
        
        return max_level
    
    def extract_skills_by_category(self, text):
        """Extract skills grouped by category"""
        text_lower = text.lower()
        skill_counts = {}
        
        for category, skills in self.skill_categories.items():
            count = sum(1 for skill in skills if skill in text_lower)
            skill_counts[f'skills_{category}'] = count
        
        skill_counts['skills_total'] = sum(skill_counts.values())
        return skill_counts
    
    def extract_job_level(self, text):
        """Determine job level from title/description"""
        text_lower = text.lower()
        
        for level, keywords in self.job_levels.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if level == 'executive':
                        return 5
                    elif level == 'senior':
                        return 4
                    elif level == 'mid':
                        return 3
                    elif level == 'junior':
                        return 2
                    elif level == 'entry':
                        return 1
        return 3  # Default to mid-level
    
    def extract_certifications_count(self, text):
        """Count certifications"""
        cert_keywords = ['certified', 'certification', 'certificate', 'aws', 'azure',
                        'google cloud', 'microsoft', 'oracle', 'cisco', 'comptia']
        text_lower = text.lower()
        return sum(1 for keyword in cert_keywords if keyword in text_lower)
    
    def extract_projects_count(self, text):
        """Estimate number of projects mentioned"""
        # Look for project section or bullet points
        project_indicators = ['project', 'developed', 'built', 'created', 'implemented',
                            'designed', 'led', 'managed']
        text_lower = text.lower()
        return min(sum(1 for indicator in project_indicators if indicator in text_lower), 20)
    
    def extract_leadership_score(self, text):
        """Score leadership experience"""
        leadership_keywords = ['lead', 'manage', 'mentor', 'supervise', 'coordinate',
                              'director', 'manager', 'head', 'team lead', 'architect']
        text_lower = text.lower()
        return min(sum(1 for keyword in leadership_keywords if keyword in text_lower), 10)
    
    def extract_all_features(self, resume_text, job_title=''):
        """Extract all features from resume"""
        combined_text = f"{resume_text} {job_title}"
        
        features = {
            'years_experience': self.extract_years_of_experience(combined_text),
            'education_level': self.extract_education_level(combined_text),
            'job_level': self.extract_job_level(combined_text),
            'certifications_count': self.extract_certifications_count(combined_text),
            'projects_count': self.extract_projects_count(combined_text),
            'leadership_score': self.extract_leadership_score(combined_text),
            'resume_length': len(resume_text),
            'word_count': len(resume_text.split())
        }
        
        # Add skill counts
        skill_features = self.extract_skills_by_category(combined_text)
        features.update(skill_features)
        
        return features
    
    def process_dataset(self, df, resume_col='Job Title', salary_col='Salary'):
        """Process entire dataset and extract features"""
        print("Extracting features from dataset...")
        
        features_list = []
        for idx, row in df.iterrows():
            resume_text = str(row[resume_col])
            features = self.extract_all_features(resume_text)
            features['salary'] = row[salary_col]
            features_list.append(features)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(df)} resumes...")
        
        features_df = pd.DataFrame(features_list)
        print(f"\n✅ Extracted {len(features_df.columns)-1} features")
        return features_df


def enhance_dataset_with_synthetic_features(df):
    """Add synthetic features to improve accuracy"""
    
    # Interaction features
    df['experience_education'] = df['years_experience'] * df['education_level']
    df['skills_per_year'] = df['skills_total'] / (df['years_experience'] + 1)
    df['leadership_experience'] = df['leadership_score'] * df['years_experience']
    
    # Polynomial features for key variables
    df['experience_squared'] = df['years_experience'] ** 2
    df['education_squared'] = df['education_level'] ** 2
    
    # Binned features
    df['experience_bin'] = pd.cut(df['years_experience'], 
                                   bins=[0, 2, 5, 10, 20, 50],
                                   labels=[1, 2, 3, 4, 5])
    df['experience_bin'] = df['experience_bin'].cat.codes + 1
    df['experience_bin'] = df['experience_bin'].fillna(3).astype(int)
    
    # Skill diversity
    skill_cols = [col for col in df.columns if col.startswith('skills_') and col != 'skills_total']
    df['skill_diversity'] = (df[skill_cols] > 0).sum(axis=1)
    
    return df


if __name__ == "__main__":
    # Test feature extraction
    extractor = ResumeFeatureExtractor()
    
    sample_resume = """
    Senior Software Engineer with 8 years of experience in Python, Java, and AWS.
    Master's degree in Computer Science. Led team of 5 developers.
    Certified AWS Solutions Architect. Built 15+ production applications.
    """
    
    features = extractor.extract_all_features(sample_resume)
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
