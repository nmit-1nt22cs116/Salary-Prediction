"""
Resume Parser Module with spaCy Integration

This module provides resume parsing functionality using spaCy NLP library
with fallback to regex-based parsing when spaCy is unavailable.
"""

import logging
import re
import signal
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import streamlit as st

# Configure logging for parser components
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resume_parser")


@dataclass
class ParserConfig:
    """
    Configuration settings for resume parser.
    
    Controls parsing behavior including spaCy model selection, fallback options,
    confidence thresholds, timeout settings, and logging preferences.
    """
    spacy_model: str = "en_core_web_sm"
    fallback_to_regex: bool = True
    min_confidence_threshold: float = 0.5
    max_parse_time_seconds: int = 5
    enable_logging: bool = True


@dataclass
class ParsedResume:
    """
    Data structure for parsed resume information.
    
    Contains all extracted fields from resume parsing including experience,
    education, demographics, job levels, and job categories. Also includes
    metadata about the parsing process such as confidence scores and method used.
    """
    # Core extracted fields
    years_exp: int = 0
    education_level: int = 3  # Default to bachelor's level
    age: int = 22  # Default age (changed to 22)
    gender: int = 1  # 0 = female/unknown, 1 = male (changed to male)
    
    # Job level flags
    is_senior: int = 0
    is_manager: int = 0
    is_executive: int = 0
    is_junior: int = 0
    is_mid_level: int = 0  # Added mid-level flag
    
    # Job category flags
    is_tech: int = 1  # Default to Technology/IT (changed to 1)
    is_sales: int = 0
    is_marketing: int = 0
    is_hr: int = 0
    
    # Metadata fields
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    parsing_method: str = "unknown"  # "spacy" or "regex"
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ParsedResume to dictionary format for backward compatibility.
        
        Returns:
            Dict containing all core fields (excluding metadata)
        """
        return {
            'years_exp': self.years_exp,
            'education_level': self.education_level,
            'age': self.age,
            'gender': self.gender,
            'is_senior': self.is_senior,
            'is_manager': self.is_manager,
            'is_executive': self.is_executive,
            'is_junior': self.is_junior,
            'is_mid_level': self.is_mid_level,
            'is_tech': self.is_tech,
            'is_sales': self.is_sales,
            'is_marketing': self.is_marketing,
            'is_hr': self.is_hr
        }


class SpacyModelManager:
    """
    Manages spaCy model loading and lifecycle with caching and error handling.
    
    This class handles the initialization and management of spaCy NLP models,
    providing graceful fallback when models are unavailable.
    """
    
    def __init__(self):
        """Initialize the SpacyModelManager with no model loaded."""
        self.nlp = None
        self.model_loaded = False
        self._model_name = None
    
    def load_model(self, model_name: str = "en_core_web_sm") -> bool:
        """
        Load spaCy model with comprehensive error handling and graceful degradation.
        
        Catches all possible errors during model loading and provides clear
        error messages for different failure scenarios. Always returns False
        on failure to enable automatic fallback to regex parsing.
        
        Args:
            model_name: Name of the spaCy model to load (default: en_core_web_sm)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self.model_loaded and self._model_name == model_name:
            logger.info(f"spaCy model '{model_name}' already loaded")
            return True
        
        try:
            logger.info(f"Attempting to load spaCy model: {model_name}")
            import spacy
            
            try:
                self.nlp = spacy.load(model_name)
                self.model_loaded = True
                self._model_name = model_name
                logger.info(f"✓ Successfully loaded spaCy model: {model_name}")
                return True
                
            except OSError as e:
                # Model not found - most common error
                logger.warning(
                    f"✗ spaCy model '{model_name}' not found. "
                    f"Falling back to regex parsing. "
                    f"To enable advanced ML extraction, install the model using: "
                    f"python -m spacy download {model_name}"
                )
                logger.debug(f"Model loading OSError details: {e}")
                self.model_loaded = False
                return False
                
        except ImportError as e:
            # spaCy not installed
            logger.warning(
                f"✗ spaCy library not installed. "
                f"Falling back to regex parsing. "
                f"To enable advanced ML extraction, install spaCy using: "
                f"pip install spacy && python -m spacy download {model_name}"
            )
            logger.debug(f"spaCy import error details: {e}")
            self.model_loaded = False
            return False
            
        except MemoryError as e:
            # Out of memory - can happen with large models
            logger.error(
                f"✗ Insufficient memory to load spaCy model '{model_name}'. "
                f"Falling back to regex parsing. "
                f"Consider using a smaller model or increasing available memory."
            )
            logger.debug(f"Memory error details: {e}")
            self.model_loaded = False
            return False
            
        except Exception as e:
            # Catch-all for any other unexpected errors
            logger.error(
                f"✗ Unexpected error loading spaCy model '{model_name}': {type(e).__name__}: {e}. "
                f"Falling back to regex parsing."
            )
            logger.debug(f"Unexpected error details: {e}", exc_info=True)
            self.model_loaded = False
            return False
    
    def is_available(self) -> bool:
        """
        Check if spaCy model is loaded and available.
        
        Returns:
            bool: True if model is loaded and ready to use, False otherwise
        """
        return self.model_loaded and self.nlp is not None
    
    def get_nlp(self):
        """
        Get the loaded spaCy model.
        
        Returns:
            spacy.Language: The loaded spaCy NLP model
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_available():
            raise RuntimeError(
                "spaCy model not loaded. Call load_model() first."
            )
        return self.nlp


@st.cache_resource
def get_spacy_model_manager() -> SpacyModelManager:
    """
    Get or create a cached SpacyModelManager instance.
    
    This function uses Streamlit's cache_resource decorator to ensure
    the spaCy model is loaded only once per session, improving performance.
    
    Returns:
        SpacyModelManager: Cached instance of SpacyModelManager
    """
    logger.info("Initializing SpacyModelManager (cached)")
    manager = SpacyModelManager()
    manager.load_model()
    return manager



class TimeoutException(Exception):
    """Exception raised when parsing operation times out."""
    pass


class ExperienceExtractor:
    """
    Extracts years of experience from resume text using spaCy NER.
    
    Uses spaCy's DATE entity recognition to identify employment periods,
    extracts job titles using ORG entities and custom patterns, and calculates
    total years of experience while handling overlapping periods.
    """
    
    def __init__(self):
        """Initialize the ExperienceExtractor."""
        self.current_year = datetime.now().year
        logger.info("ExperienceExtractor initialized")
    
    def extract_years(self, doc) -> tuple[int, float]:
        """
        Extract total years of experience with confidence score.
        
        Args:
            doc: spaCy Doc object containing processed resume text
            
        Returns:
            Tuple of (years_of_experience, confidence_score)
        """
        # Find all date ranges in the document
        date_ranges = self._find_date_ranges(doc)
        
        # Extract job titles for context
        job_titles = self._extract_job_titles(doc)
        
        # Calculate total years from date ranges
        total_years = self._calculate_total_years(date_ranges)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(date_ranges, job_titles, doc)
        
        logger.info(
            f"Extracted {total_years} years of experience "
            f"(confidence: {confidence:.2f}) from {len(date_ranges)} date ranges"
        )
        
        return total_years, confidence
    
    def _find_date_ranges(self, doc) -> List[tuple[int, int]]:
        """
        Find employment date ranges from resume text.
        
        Identifies date ranges in formats:
        - "YYYY - YYYY"
        - "YYYY - Present"
        - "Month YYYY - Month YYYY"
        - "MM/YYYY - MM/YYYY"
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of tuples containing (start_year, end_year)
        """
        date_ranges = []
        text = doc.text
        
        # Pattern 1: YYYY - YYYY or YYYY - Present (most common)
        pattern1 = r'(\d{4})\s*[-–—to]+\s*(?:(\d{4})|present|current|now|till\s+date|ongoing)'
        matches1 = re.finditer(pattern1, text, re.IGNORECASE)
        
        for match in matches1:
            start_year = int(match.group(1))
            end_str = match.group(2)
            end_year = int(end_str) if end_str else self.current_year
            
            # Validate years are reasonable
            if 1980 <= start_year <= end_year <= self.current_year + 1:
                date_ranges.append((start_year, end_year))
                logger.debug(f"Found date range: {start_year} - {end_year}")
        
        # Pattern 2: Month YYYY - Month YYYY
        months = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
        pattern2 = rf'{months}\s+(\d{{4}})\s*[-–—to]+\s*(?:{months}\s+(\d{{4}})|present|current|now|till\s+date|ongoing)'
        matches2 = re.finditer(pattern2, text, re.IGNORECASE)
        
        for match in matches2:
            start_year = int(match.group(1))
            end_str = match.group(2)
            end_year = int(end_str) if end_str else self.current_year
            
            # Validate years are reasonable
            if 1980 <= start_year <= end_year <= self.current_year + 1:
                # Check if this range is not already captured
                if (start_year, end_year) not in date_ranges:
                    date_ranges.append((start_year, end_year))
                    logger.debug(f"Found date range with months: {start_year} - {end_year}")
        
        # Pattern 3: MM/YYYY - MM/YYYY format
        pattern3 = r'(\d{1,2})/(\d{4})\s*[-–—to]+\s*(?:(\d{1,2})/(\d{4})|present|current|now)'
        matches3 = re.finditer(pattern3, text, re.IGNORECASE)
        
        for match in matches3:
            start_year = int(match.group(2))
            end_str = match.group(4)
            end_year = int(end_str) if end_str else self.current_year
            
            # Validate years are reasonable
            if 1980 <= start_year <= end_year <= self.current_year + 1:
                if (start_year, end_year) not in date_ranges:
                    date_ranges.append((start_year, end_year))
                    logger.debug(f"Found date range MM/YYYY format: {start_year} - {end_year}")
        
        # Pattern 4: "X years of experience" or "X+ years experience"
        exp_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)'
        exp_matches = re.finditer(exp_pattern, text, re.IGNORECASE)
        
        for match in exp_matches:
            years = int(match.group(1))
            if 0 < years <= 50:  # Reasonable range
                # Convert to date range (assume current position)
                start_year = self.current_year - years
                end_year = self.current_year
                if (start_year, end_year) not in date_ranges:
                    date_ranges.append((start_year, end_year))
                    logger.debug(f"Found experience statement: {years} years -> {start_year} - {end_year}")
        
        # Also use spaCy's DATE entities as additional hints
        for ent in doc.ents:
            if ent.label_ == "DATE":
                # Try to extract year from DATE entity
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', ent.text)
                if year_match:
                    year = int(year_match.group(1))
                    logger.debug(f"Found DATE entity with year: {year}")
        
        return date_ranges
    
    def _extract_job_titles(self, doc) -> List[str]:
        """
        Extract job titles using ORG entities and custom patterns.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of identified job titles
        """
        job_titles = []
        
        # Common job title keywords
        title_keywords = [
            'engineer', 'developer', 'manager', 'director', 'analyst',
            'consultant', 'specialist', 'coordinator', 'lead', 'architect',
            'designer', 'scientist', 'administrator', 'officer', 'executive',
            'associate', 'assistant', 'supervisor', 'technician', 'intern'
        ]
        
        # Look for job title patterns in text
        text_lower = doc.text.lower()
        for keyword in title_keywords:
            # Find sentences containing job title keywords
            pattern = rf'\b\w+\s+{keyword}\b'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                title = match.group(0)
                if title not in job_titles:
                    job_titles.append(title)
                    logger.debug(f"Found job title: {title}")
        
        # Also extract ORG entities as potential employers
        for ent in doc.ents:
            if ent.label_ == "ORG":
                logger.debug(f"Found organization: {ent.text}")
        
        return job_titles
    
    def _calculate_total_years(self, date_ranges: List[tuple[int, int]]) -> int:
        """
        Calculate total years from date ranges, handling overlapping periods.
        
        Args:
            date_ranges: List of (start_year, end_year) tuples
            
        Returns:
            Total years of experience
        """
        if not date_ranges:
            return 0
        
        # Sort date ranges by start year
        sorted_ranges = sorted(date_ranges)
        
        # Merge overlapping ranges
        merged_ranges = []
        current_start, current_end = sorted_ranges[0]
        
        for start, end in sorted_ranges[1:]:
            if start <= current_end:
                # Overlapping or adjacent - merge
                current_end = max(current_end, end)
            else:
                # Non-overlapping - save current and start new
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add the last range
        merged_ranges.append((current_start, current_end))
        
        # Calculate total years from merged ranges
        total_years = sum(end - start for start, end in merged_ranges)
        
        logger.debug(
            f"Calculated {total_years} years from {len(date_ranges)} ranges "
            f"(merged to {len(merged_ranges)} ranges)"
        )
        
        return total_years
    
    def _calculate_confidence(
        self, 
        date_ranges: List[tuple[int, int]], 
        job_titles: List[str],
        doc
    ) -> float:
        """
        Calculate confidence score for experience extraction.
        
        Confidence is based on:
        - Number of date ranges found (more is better)
        - Presence of job titles near dates (boosts confidence)
        - Completeness of dates (reduces confidence for ambiguous dates)
        
        Args:
            date_ranges: List of extracted date ranges
            job_titles: List of extracted job titles
            doc: spaCy Doc object
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence starts at 0.3
        confidence = 0.3
        
        # Boost confidence based on number of date ranges found
        if len(date_ranges) >= 3:
            confidence += 0.4
        elif len(date_ranges) == 2:
            confidence += 0.3
        elif len(date_ranges) == 1:
            confidence += 0.2
        else:
            # No date ranges found - very low confidence
            return 0.2
        
        # Boost confidence if job titles are present
        if len(job_titles) >= 2:
            confidence += 0.2
        elif len(job_titles) == 1:
            confidence += 0.1
        
        # Check for proximity of job titles to dates
        # If job titles and dates appear near each other, boost confidence
        text = doc.text.lower()
        dates_near_titles = 0
        
        for title in job_titles:
            # Look for dates within 100 characters of job title
            title_pos = text.find(title.lower())
            if title_pos != -1:
                context = text[max(0, title_pos - 100):min(len(text), title_pos + 100)]
                if re.search(r'\d{4}', context):
                    dates_near_titles += 1
        
        if dates_near_titles > 0:
            confidence += 0.1
        
        # Reduce confidence for incomplete or ambiguous dates
        # Check if we have "Present" or "Current" in multiple places (might indicate ongoing roles)
        present_count = len(re.findall(r'\b(?:present|current|now)\b', text, re.IGNORECASE))
        if present_count > 2:
            # Multiple "present" entries might indicate confusion
            confidence -= 0.1
        
        # Cap confidence at 1.0
        confidence = min(1.0, confidence)
        
        return confidence


class EducationExtractor:
    """
    Extracts education level from resume text using spaCy pattern matching.
    
    Uses spaCy Matcher to identify degree patterns and abbreviations,
    classifies education into 5 levels, and returns the highest degree found
    with a confidence score.
    """
    
    def __init__(self):
        """Initialize the EducationExtractor with degree patterns and levels."""
        # Define degree levels mapping (phd=5, master=4, bachelor=3, diploma=2, high school=1)
        self.degree_levels = {
            # Doctorate level (5)
            'phd': 5, 'ph.d': 5, 'ph.d.': 5, 'doctorate': 5, 'd.phil': 5, 'd.phil.': 5,
            'doctor of philosophy': 5, 'doctoral': 5,
            
            # Master's level (4)
            'mba': 4, 'master': 4, 'masters': 4, "master's": 4,
            'ms': 4, 'm.s': 4, 'm.s.': 4,
            'm.tech': 4, 'm.tech.': 4, 'mtech': 4,
            'm.sc': 4, 'm.sc.': 4, 'msc': 4,
            'ma': 4, 'm.a': 4, 'm.a.': 4,
            'master of science': 4, 'master of arts': 4, 'master of business': 4,
            'master of engineering': 4, 'master of technology': 4,
            
            # Bachelor's level (3)
            'bachelor': 3, 'bachelors': 3, "bachelor's": 3,
            'bs': 3, 'b.s': 3, 'b.s.': 3,
            'ba': 3, 'b.a': 3, 'b.a.': 3,
            'b.tech': 3, 'b.tech.': 3, 'btech': 3,
            'be': 3, 'b.e': 3, 'b.e.': 3,
            'b.sc': 3, 'b.sc.': 3, 'bsc': 3,
            'bachelor of science': 3, 'bachelor of arts': 3,
            'bachelor of engineering': 3, 'bachelor of technology': 3,
            
            # Diploma level (2)
            'diploma': 2, 'associate': 2, 'associates': 2, "associate's": 2,
            'associate degree': 2, 'advanced diploma': 2,
            
            # High school level (1)
            'high school': 1, 'secondary': 1, 'secondary school': 1,
            'high school diploma': 1, 'ged': 1, 'hssc': 1, 'hsc': 1
        }
        
        logger.info("EducationExtractor initialized with degree patterns")
    
    def extract_education(self, doc) -> tuple[int, float]:
        """
        Extract highest education level with confidence score.
        
        Args:
            doc: spaCy Doc object containing processed resume text
            
        Returns:
            Tuple of (education_level, confidence_score)
            education_level: 1-5 (high school to doctorate)
            confidence_score: 0.0-1.0
        """
        text = doc.text.lower()
        
        # Find all degree matches
        found_degrees = []
        found_patterns = []
        
        # Search for all degree keywords in text
        for degree_keyword, level in self.degree_levels.items():
            # Use word boundaries to avoid partial matches
            pattern = rf'\b{re.escape(degree_keyword)}\b'
            if re.search(pattern, text):
                found_degrees.append(level)
                found_patterns.append(degree_keyword)
                logger.debug(f"Found degree pattern: {degree_keyword} (level {level})")
        
        # If no degrees found, return default bachelor's level with low confidence
        if not found_degrees:
            logger.info("No education patterns found, returning default level 3")
            return 3, 0.3
        
        # Get highest degree level
        highest_level = max(found_degrees)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            highest_level, 
            found_degrees, 
            found_patterns, 
            text
        )
        
        logger.info(
            f"Extracted education level {highest_level} "
            f"(confidence: {confidence:.2f}) from {len(found_degrees)} matches"
        )
        
        return highest_level, confidence
    
    def _build_degree_patterns(self):
        """
        Build spaCy matcher patterns for common degrees and abbreviations.
        
        This method creates patterns for the spaCy Matcher to identify
        degree mentions in text. Currently using regex for simplicity,
        but can be enhanced with spaCy Matcher for more complex patterns.
        
        Returns:
            List of spaCy Matcher patterns
        """
        # This method is kept for future enhancement with spaCy Matcher
        # Currently using regex-based matching in extract_education()
        patterns = []
        
        # Example pattern structure for future spaCy Matcher integration:
        # patterns.append([{"LOWER": "bachelor"}, {"LOWER": "of"}, {"LOWER": "science"}])
        # patterns.append([{"LOWER": "b"}, {"TEXT": "."}, {"LOWER": "s"}])
        
        return patterns
    
    def _calculate_confidence(
        self,
        highest_level: int,
        found_degrees: List[int],
        found_patterns: List[str],
        text: str
    ) -> float:
        """
        Calculate confidence score for education extraction.
        
        Confidence is based on:
        - Number of degree mentions (more is better, but diminishing returns)
        - Pattern match strength (full names vs abbreviations)
        - Presence of education section headers
        - Consistency of degree levels found
        
        Args:
            highest_level: The highest education level found
            found_degrees: List of all degree levels found
            found_patterns: List of all degree patterns matched
            text: Resume text (lowercase)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.5
        
        # Boost confidence based on number of matches
        num_matches = len(found_degrees)
        if num_matches >= 3:
            confidence += 0.2
        elif num_matches == 2:
            confidence += 0.15
        elif num_matches == 1:
            confidence += 0.1
        
        # Boost confidence for full degree names vs abbreviations
        full_names = [
            'bachelor of science', 'bachelor of arts', 'bachelor of engineering',
            'master of science', 'master of arts', 'master of business',
            'doctor of philosophy', 'doctorate'
        ]
        has_full_name = any(name in text for name in full_names)
        if has_full_name:
            confidence += 0.15
        
        # Boost confidence if education section header is present
        education_headers = [
            'education', 'academic', 'qualification', 'degree',
            'educational background', 'academic background'
        ]
        has_education_section = any(header in text for header in education_headers)
        if has_education_section:
            confidence += 0.1
        
        # Check consistency - if all found degrees are at the same level or close
        if found_degrees:
            level_variance = max(found_degrees) - min(found_degrees)
            if level_variance == 0:
                # All degrees at same level - very consistent
                confidence += 0.1
            elif level_variance == 1:
                # Close levels (e.g., bachelor and master) - still good
                confidence += 0.05
        
        # Reduce confidence for very short patterns (might be false positives)
        short_patterns = [p for p in found_patterns if len(p) <= 3]
        if len(short_patterns) == len(found_patterns) and len(short_patterns) > 0:
            # All matches are short abbreviations - slightly less confident
            confidence -= 0.05
        
        # Cap confidence at 1.0
        confidence = min(1.0, max(0.3, confidence))
        
        return confidence


@contextmanager
def timeout_context(seconds: int):
    """
    Context manager for timeout protection on Windows.
    
    Note: signal.alarm is not available on Windows, so this provides
    a basic timeout mechanism that logs warnings but doesn't enforce hard timeouts.
    For production use on Unix systems, this can be enhanced with signal.SIGALRM.
    
    Args:
        seconds: Maximum time allowed for operation
    """
    # On Windows, signal.alarm is not available
    # We'll implement a simple logging-based approach
    import time
    start_time = time.time()
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if elapsed > seconds:
            logger.warning(f"Operation took {elapsed:.2f}s, exceeding timeout of {seconds}s")


class SkillsExtractor:
    """
    Extracts and categorizes technical and professional skills from resume text.
    
    Uses spaCy PhraseMatcher for efficient skill matching across a comprehensive
    skills database. Categorizes skills into domains (programming, data science,
    cloud, frameworks, databases) and sets job category flags.
    """
    
    def __init__(self):
        """Initialize the SkillsExtractor with skills database."""
        self.tech_skills = self._load_tech_skills()
        self.skill_categories = self._build_skill_categories()
        logger.info(f"SkillsExtractor initialized with {len(self.tech_skills)} skills")
    
    def extract_skills(self, doc) -> tuple[List[str], Dict[str, int], float]:
        """
        Extract skills and categorize them.
        
        Args:
            doc: spaCy Doc object containing processed resume text
            
        Returns:
            Tuple of (skills_list, category_flags, confidence_score)
            skills_list: List of identified skills
            category_flags: Dict with is_tech, is_sales, is_marketing, is_hr flags
            confidence_score: 0.0-1.0
        """
        text_lower = doc.text.lower()
        
        # Find all skills in the text
        found_skills = []
        skill_positions = []
        
        # Match skills case-insensitively
        for skill in self.tech_skills:
            # Use word boundaries to avoid partial matches
            pattern = rf'\b{re.escape(skill.lower())}\b'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if skill not in found_skills:
                    found_skills.append(skill)
                    skill_positions.append(match.start())
                    logger.debug(f"Found skill: {skill}")
        
        # Categorize the found skills
        category_counts = self._categorize_skills(found_skills)
        
        # Set job category flags based on skills found
        category_flags = {
            'is_tech': 1 if category_counts.get('total_tech', 0) > 0 else 0,
            'is_sales': 0,  # Will be set based on sales keywords
            'is_marketing': 0,  # Will be set based on marketing keywords
            'is_hr': 0  # Will be set based on HR keywords
        }
        
        # Check for non-tech job categories
        sales_keywords = ['sales', 'account management', 'business development', 'crm', 'salesforce']
        marketing_keywords = ['marketing', 'seo', 'sem', 'social media', 'content marketing', 'brand']
        hr_keywords = ['hr', 'human resources', 'recruitment', 'talent acquisition', 'hiring']
        
        for keyword in sales_keywords:
            if keyword in text_lower:
                category_flags['is_sales'] = 1
                break
        
        for keyword in marketing_keywords:
            if keyword in text_lower:
                category_flags['is_marketing'] = 1
                break
        
        for keyword in hr_keywords:
            if keyword in text_lower:
                category_flags['is_hr'] = 1
                break
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            found_skills,
            category_counts,
            text_lower
        )
        
        logger.info(
            f"Extracted {len(found_skills)} skills "
            f"(confidence: {confidence:.2f}), "
            f"tech: {category_flags['is_tech']}, "
            f"categories: {category_counts}"
        )
        
        return found_skills, category_flags, confidence
    
    def _load_tech_skills(self) -> set:
        """
        Load technical skills database with 100+ skills.
        
        Returns:
            Set of technical skills covering programming, data science,
            cloud, frameworks, and databases
        """
        skills = set()
        
        # Programming languages (30+)
        programming = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c',
            'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala',
            'r', 'matlab', 'perl', 'shell', 'bash', 'powershell',
            'objective-c', 'dart', 'lua', 'haskell', 'elixir', 'clojure',
            'groovy', 'f#', 'vb.net', 'assembly', 'cobol', 'fortran'
        ]
        skills.update(programming)
        
        # Data Science & ML (25+)
        data_science = [
            'machine learning', 'deep learning', 'neural networks',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
            'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
            'data analysis', 'data science', 'statistics', 'nlp',
            'computer vision', 'ai', 'artificial intelligence',
            'data mining', 'big data', 'hadoop', 'spark', 'kafka',
            'tableau', 'power bi'
        ]
        skills.update(data_science)
        
        # Cloud & DevOps (20+)
        cloud = [
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
            'jenkins', 'ci/cd', 'terraform', 'ansible', 'chef', 'puppet',
            'cloudformation', 'lambda', 'ec2', 's3', 'ecs', 'eks',
            'devops', 'microservices', 'serverless'
        ]
        skills.update(cloud)
        
        # Web Frameworks & Libraries (25+)
        frameworks = [
            'react', 'angular', 'vue', 'vue.js', 'node.js', 'express',
            'django', 'flask', 'fastapi', 'spring', 'spring boot',
            'asp.net', '.net', 'rails', 'laravel', 'symfony',
            'next.js', 'nuxt.js', 'svelte', 'ember', 'backbone',
            'jquery', 'bootstrap', 'tailwind', 'material-ui'
        ]
        skills.update(frameworks)
        
        # Databases (15+)
        databases = [
            'sql', 'mysql', 'postgresql', 'oracle', 'sql server',
            'mongodb', 'cassandra', 'redis', 'elasticsearch',
            'dynamodb', 'neo4j', 'couchdb', 'sqlite', 'mariadb',
            'firebase'
        ]
        skills.update(databases)
        
        # Additional technical skills (15+)
        other_tech = [
            'git', 'github', 'gitlab', 'bitbucket', 'jira', 'agile',
            'scrum', 'rest api', 'graphql', 'websockets', 'oauth',
            'jwt', 'linux', 'unix', 'windows server'
        ]
        skills.update(other_tech)
        
        logger.debug(f"Loaded {len(skills)} technical skills")
        return skills
    
    def _build_skill_categories(self) -> Dict[str, List[str]]:
        """
        Build skill category mappings for classification.
        
        Returns:
            Dictionary mapping category names to lists of skills
        """
        categories = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c',
                'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala',
                'r', 'matlab', 'perl', 'shell', 'bash'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch',
                'keras', 'scikit-learn', 'pandas', 'numpy', 'data analysis',
                'data science', 'statistics', 'nlp', 'ai', 'big data',
                'hadoop', 'spark', 'tableau', 'power bi'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
                'jenkins', 'ci/cd', 'terraform', 'ansible', 'devops',
                'microservices', 'serverless', 'lambda', 'ec2', 's3'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'node.js', 'express', 'django',
                'flask', 'spring', 'asp.net', '.net', 'rails', 'laravel',
                'next.js', 'bootstrap', 'tailwind'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
                'elasticsearch', 'oracle', 'cassandra', 'dynamodb', 'neo4j'
            ]
        }
        
        return categories
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, int]:
        """
        Categorize skills into domains and count them.
        
        Args:
            skills: List of identified skills
            
        Returns:
            Dictionary with counts for each category
        """
        category_counts = {
            'programming': 0,
            'data_science': 0,
            'cloud': 0,
            'frameworks': 0,
            'databases': 0,
            'total_tech': 0
        }
        
        # Count skills in each category
        for skill in skills:
            skill_lower = skill.lower()
            categorized = False
            
            for category, category_skills in self.skill_categories.items():
                if skill_lower in category_skills:
                    category_counts[category] += 1
                    categorized = True
            
            # Count as technical skill if it's in our database
            if skill_lower in self.tech_skills:
                category_counts['total_tech'] += 1
        
        logger.debug(f"Skill categorization: {category_counts}")
        return category_counts
    
    def _calculate_confidence(
        self,
        found_skills: List[str],
        category_counts: Dict[str, int],
        text: str
    ) -> float:
        """
        Calculate confidence score for skills extraction.
        
        Confidence is based on:
        - Number of skills found (more is better)
        - Presence of skills section header
        - Diversity of skill categories
        - Context around skills (in job descriptions vs random mentions)
        
        Args:
            found_skills: List of identified skills
            category_counts: Dictionary with category counts
            text: Resume text (lowercase)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.4
        
        # Boost confidence based on number of skills found
        num_skills = len(found_skills)
        if num_skills >= 10:
            confidence += 0.3
        elif num_skills >= 5:
            confidence += 0.2
        elif num_skills >= 2:
            confidence += 0.1
        elif num_skills == 0:
            # No skills found - low confidence
            return 0.3
        
        # Boost confidence if skills section is present
        skills_headers = [
            'skills', 'technical skills', 'core competencies',
            'technologies', 'expertise', 'proficiencies'
        ]
        has_skills_section = any(header in text for header in skills_headers)
        if has_skills_section:
            confidence += 0.2
        
        # Boost confidence for diverse skill categories
        categories_with_skills = sum(1 for count in category_counts.values() if count > 0)
        if categories_with_skills >= 4:
            confidence += 0.15
        elif categories_with_skills >= 3:
            confidence += 0.1
        elif categories_with_skills >= 2:
            confidence += 0.05
        
        # Check if skills appear in context (near job descriptions or projects)
        context_keywords = [
            'experience', 'project', 'developed', 'built', 'implemented',
            'worked with', 'using', 'utilized'
        ]
        has_context = any(keyword in text for keyword in context_keywords)
        if has_context:
            confidence += 0.1
        
        # Cap confidence at 1.0
        confidence = min(1.0, confidence)
        
        return confidence


class JobLevelClassifier:
    """
    Classifies job seniority level from resume text using spaCy pattern matching.
    
    Analyzes job titles and responsibilities to determine seniority level,
    setting flags for junior, senior, manager, and executive positions.
    Uses spaCy Matcher for keyword patterns and responsibility analysis.
    """
    
    def __init__(self):
        """Initialize the JobLevelClassifier with level patterns."""
        self.level_patterns = self._build_level_patterns()
        logger.info("JobLevelClassifier initialized with level patterns")
    
    def classify_level(self, doc, job_titles: List[str] = None) -> tuple[Dict[str, int], float]:
        """
        Classify job level from titles and context.
        
        Args:
            doc: spaCy Doc object containing processed resume text
            job_titles: Optional list of extracted job titles
            
        Returns:
            Tuple of (level_flags, confidence_score)
            level_flags: Dict with is_junior, is_senior, is_manager, is_executive flags
            confidence_score: 0.0-1.0
        """
        text_lower = doc.text.lower()
        
        # Initialize level flags
        level_flags = {
            'is_junior': 0,
            'is_senior': 0,
            'is_manager': 0,
            'is_executive': 0
        }
        
        # Track matches for confidence calculation
        matches_found = {
            'junior': [],
            'senior': [],
            'manager': [],
            'executive': []
        }
        
        # Search for level keywords in text
        for level, patterns in self.level_patterns.items():
            for pattern in patterns:
                # Use word boundaries to avoid partial matches
                regex_pattern = rf'\b{re.escape(pattern)}\b'
                matches = re.finditer(regex_pattern, text_lower)
                for match in matches:
                    matches_found[level].append(match.group(0))
                    logger.debug(f"Found {level} pattern: {match.group(0)}")
        
        # Set flags based on matches found
        if matches_found['junior']:
            level_flags['is_junior'] = 1
        if matches_found['senior']:
            level_flags['is_senior'] = 1
        if matches_found['manager']:
            level_flags['is_manager'] = 1
        if matches_found['executive']:
            level_flags['is_executive'] = 1
        
        # Analyze responsibilities for additional level inference
        responsibility_scores = self._analyze_responsibilities(doc)
        
        # Boost manager/executive flags based on responsibilities
        if responsibility_scores['manager'] > 0.5 and not level_flags['is_manager']:
            level_flags['is_manager'] = 1
            logger.debug("Set is_manager flag based on responsibility analysis")
        
        if responsibility_scores['executive'] > 0.5 and not level_flags['is_executive']:
            level_flags['is_executive'] = 1
            logger.debug("Set is_executive flag based on responsibility analysis")
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            matches_found,
            responsibility_scores,
            text_lower
        )
        
        logger.info(
            f"Classified job level: junior={level_flags['is_junior']}, "
            f"senior={level_flags['is_senior']}, manager={level_flags['is_manager']}, "
            f"executive={level_flags['is_executive']} (confidence: {confidence:.2f})"
        )
        
        return level_flags, confidence
    
    def _build_level_patterns(self) -> Dict[str, List[str]]:
        """
        Build patterns for each job level using keywords.
        
        Returns:
            Dictionary mapping level names to lists of keyword patterns
        """
        patterns = {
            'junior': [
                'junior', 'jr', 'jr.', 'entry level', 'entry-level',
                'associate', 'intern', 'trainee', 'graduate'
            ],
            'senior': [
                'senior', 'sr', 'sr.', 'lead', 'principal', 'staff',
                'expert', 'specialist', 'advanced', 'chief'
            ],
            'manager': [
                'manager', 'mgr', 'supervisor', 'team lead', 'team leader',
                'head', 'coordinator', 'project manager', 'program manager',
                'product manager', 'engineering manager', 'technical lead'
            ],
            'executive': [
                'director', 'vp', 'vice president', 'ceo', 'cto', 'cfo',
                'coo', 'president', 'executive', 'c-level', 'founder',
                'co-founder', 'partner', 'principal', 'head of'
            ]
        }
        
        logger.debug(f"Built level patterns: {sum(len(v) for v in patterns.values())} total patterns")
        return patterns
    
    def _analyze_responsibilities(self, doc) -> Dict[str, float]:
        """
        Analyze job responsibilities for level indicators.
        
        Searches for leadership keywords like "managed", "led", "directed",
        "supervised", "mentored" to infer seniority when titles are ambiguous.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Dictionary with responsibility scores for manager and executive levels
        """
        text_lower = doc.text.lower()
        
        # Leadership keywords for manager level
        manager_keywords = [
            'managed', 'manage', 'managing', 'supervised', 'supervise',
            'led', 'lead', 'leading', 'coordinated', 'coordinate',
            'mentored', 'mentor', 'trained', 'train', 'guided',
            'oversaw', 'oversee', 'team of'
        ]
        
        # Strategic/executive keywords for executive level
        executive_keywords = [
            'directed', 'direct', 'directing', 'established', 'establish',
            'founded', 'found', 'built', 'grew', 'scaled', 'transformed',
            'strategic', 'strategy', 'vision', 'roadmap', 'p&l',
            'budget', 'revenue', 'organization', 'company-wide',
            'cross-functional', 'stakeholder'
        ]
        
        # Count keyword occurrences
        manager_count = 0
        executive_count = 0
        
        for keyword in manager_keywords:
            # Use word boundaries to avoid partial matches
            pattern = rf'\b{re.escape(keyword)}\b'
            matches = re.findall(pattern, text_lower)
            manager_count += len(matches)
            if matches:
                logger.debug(f"Found manager responsibility keyword: {keyword} ({len(matches)} times)")
        
        for keyword in executive_keywords:
            pattern = rf'\b{re.escape(keyword)}\b'
            matches = re.findall(pattern, text_lower)
            executive_count += len(matches)
            if matches:
                logger.debug(f"Found executive responsibility keyword: {keyword} ({len(matches)} times)")
        
        # Calculate scores (normalize to 0-1 range)
        # More than 3 manager keywords = high confidence
        # More than 5 executive keywords = high confidence
        manager_score = min(1.0, manager_count / 3.0)
        executive_score = min(1.0, executive_count / 5.0)
        
        logger.debug(
            f"Responsibility analysis: manager_score={manager_score:.2f} "
            f"({manager_count} keywords), executive_score={executive_score:.2f} "
            f"({executive_count} keywords)"
        )
        
        return {
            'manager': manager_score,
            'executive': executive_score
        }
    
    def _calculate_confidence(
        self,
        matches_found: Dict[str, List[str]],
        responsibility_scores: Dict[str, float],
        text: str
    ) -> float:
        """
        Calculate confidence score for job level classification.
        
        Confidence is based on:
        - Number and clarity of level keywords found
        - Presence of responsibility indicators
        - Context around level keywords (in job titles vs random mentions)
        - Consistency of level indicators
        
        Args:
            matches_found: Dictionary of level matches found
            responsibility_scores: Dictionary of responsibility analysis scores
            text: Resume text (lowercase)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.4
        
        # Count total matches across all levels
        total_matches = sum(len(matches) for matches in matches_found.values())
        
        # Boost confidence based on number of level indicators found
        if total_matches >= 5:
            confidence += 0.3
        elif total_matches >= 3:
            confidence += 0.2
        elif total_matches >= 1:
            confidence += 0.1
        else:
            # No level indicators found - low confidence
            return 0.3
        
        # Boost confidence if responsibility keywords are present
        if responsibility_scores['manager'] > 0.3 or responsibility_scores['executive'] > 0.3:
            confidence += 0.15
        
        # Check for job title context
        # Level keywords near "title", "position", "role" are more reliable
        title_context_keywords = ['title', 'position', 'role', 'as a', 'as an']
        has_title_context = any(keyword in text for keyword in title_context_keywords)
        if has_title_context:
            confidence += 0.1
        
        # Check for experience section
        experience_headers = [
            'experience', 'work history', 'employment', 'professional experience',
            'career history', 'work experience'
        ]
        has_experience_section = any(header in text for header in experience_headers)
        if has_experience_section:
            confidence += 0.1
        
        # Reduce confidence if conflicting levels are found
        # (e.g., both junior and executive - unusual)
        levels_found = sum(1 for matches in matches_found.values() if matches)
        if levels_found >= 3:
            # Multiple different levels found - might be career progression or confusion
            confidence -= 0.1
        
        # Check for most recent position indicators
        recent_indicators = ['current', 'present', 'currently', 'latest', 'most recent']
        has_recent_indicator = any(indicator in text for indicator in recent_indicators)
        if has_recent_indicator:
            # Having recent position indicator helps with accuracy
            confidence += 0.05
        
        # Cap confidence at 1.0
        confidence = min(1.0, max(0.3, confidence))
        
        return confidence


class ResumeParser:
    """
    Main resume parser with dual parsing strategy.
    
    Attempts to parse resumes using spaCy NLP first for better accuracy,
    with automatic fallback to regex-based parsing when spaCy is unavailable
    or encounters errors.
    """
    
    def __init__(
        self, 
        spacy_manager: SpacyModelManager,
        config: Optional[ParserConfig] = None
    ):
        """
        Initialize ResumeParser with spaCy manager and configuration.
        
        Args:
            spacy_manager: SpacyModelManager instance for NLP operations
            config: Optional ParserConfig for customizing parser behavior
        """
        self.spacy_manager = spacy_manager
        self.config = config or ParserConfig()
        
        if self.config.enable_logging:
            logger.info("ResumeParser initialized with dual parsing strategy")
    
    def parse(self, text: str) -> ParsedResume:
        """
        Parse resume text and return structured data with graceful degradation.
        
        Uses spaCy-first strategy with automatic fallback to regex parsing.
        Includes timeout protection and comprehensive error handling.
        All errors are caught and logged, ensuring the parser never fails completely.
        
        Args:
            text: Raw resume text to parse
            
        Returns:
            ParsedResume: Structured resume data with confidence scores
        """
        import time
        parse_start_time = time.time()
        
        # Validate input
        if not text or len(text) < 100:
            logger.warning(
                "✗ Resume text too short or empty (length: {0}). "
                "Returning default values.".format(len(text) if text else 0)
            )
            result = ParsedResume(parsing_method="invalid_input")
            return result
        
        logger.info(f"Starting resume parsing | Text length: {len(text)} characters")
        
        # Try spaCy parsing first if available
        if self.spacy_manager.is_available():
            try:
                logger.info("→ Attempting spaCy-based parsing (advanced ML extraction)")
                with timeout_context(self.config.max_parse_time_seconds):
                    result = self._parse_with_spacy(text)
                    result.parsing_method = "spacy"
                    
                    # Log final summary
                    total_parse_time = time.time() - parse_start_time
                    logger.info(
                        f"✓ Resume parsing completed successfully | "
                        f"Method: spacy | "
                        f"Overall confidence: {result.confidence_scores.get('overall', 0.0):.2f} | "
                        f"Total time: {total_parse_time:.3f}s"
                    )
                    return result
                    
            except TimeoutException:
                logger.warning(
                    f"✗ spaCy parsing timed out after {self.config.max_parse_time_seconds}s. "
                    f"Falling back to regex parsing for faster results."
                )
                if not self.config.fallback_to_regex:
                    raise
                    
            except MemoryError as e:
                logger.error(
                    f"✗ Insufficient memory for spaCy parsing. "
                    f"Falling back to regex parsing (lower memory usage)."
                )
                logger.debug(f"Memory error details: {e}")
                if not self.config.fallback_to_regex:
                    raise
                    
            except Exception as e:
                logger.warning(
                    f"✗ spaCy parsing failed with {type(e).__name__}: {e}. "
                    f"Falling back to regex parsing."
                )
                logger.debug(f"spaCy parsing error details: {e}", exc_info=True)
                if not self.config.fallback_to_regex:
                    raise
        else:
            logger.info(
                "→ spaCy not available. Using regex-based parsing. "
                "Install spaCy for improved accuracy: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
        
        # Fallback to regex parsing
        try:
            logger.info("→ Using regex-based parsing (pattern matching)")
            result = self._parse_with_regex(text)
            result.parsing_method = "regex"
            
            # Calculate overall confidence for regex parsing
            overall_confidence = self._calculate_overall_confidence(result.confidence_scores)
            result.confidence_scores['overall'] = overall_confidence
            
            # Log final summary for regex parsing
            total_parse_time = time.time() - parse_start_time
            logger.info(
                f"✓ Resume parsing completed | "
                f"Method: regex | "
                f"Overall confidence: {overall_confidence:.2f} | "
                f"Total time: {total_parse_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            # Last resort: return default values if even regex parsing fails
            logger.error(
                f"✗ Critical error: Both spaCy and regex parsing failed. "
                f"Returning default values. Error: {type(e).__name__}: {e}"
            )
            logger.debug(f"Regex parsing error details: {e}", exc_info=True)
            
            result = ParsedResume(parsing_method="error")
            result.confidence_scores['overall'] = 0.1
            
            total_parse_time = time.time() - parse_start_time
            logger.error(
                f"✗ Resume parsing failed | "
                f"Method: error | "
                f"Returning default values | "
                f"Total time: {total_parse_time:.3f}s"
            )
            
            return result
    
    def _parse_with_spacy(self, text: str) -> ParsedResume:
        """
        Parse resume using spaCy NLP and extractors.
        
        Uses spaCy NLP to process the resume text and applies specialized
        extractors for experience, education, skills, and job level classification.
        
        Args:
            text: Resume text to parse
            
        Returns:
            ParsedResume: Structured resume data
        """
        import time
        start_time = time.time()
        
        # Get spaCy NLP model
        nlp = self.spacy_manager.get_nlp()
        
        # Process text with spaCy
        doc = nlp(text)
        nlp_time = time.time() - start_time
        logger.debug(f"spaCy NLP processing completed in {nlp_time:.3f}s")
        
        # Initialize result
        result = ParsedResume()
        
        # Extract experience using ExperienceExtractor
        exp_start = time.time()
        experience_extractor = ExperienceExtractor()
        years_exp, exp_confidence = experience_extractor.extract_years(doc)
        result.years_exp = years_exp
        result.confidence_scores['years_exp'] = exp_confidence
        exp_time = time.time() - exp_start
        logger.debug(f"Experience extraction completed in {exp_time:.3f}s")
        
        # Extract education using EducationExtractor
        edu_start = time.time()
        education_extractor = EducationExtractor()
        education_level, edu_confidence = education_extractor.extract_education(doc)
        result.education_level = education_level
        result.confidence_scores['education_level'] = edu_confidence
        edu_time = time.time() - edu_start
        logger.debug(f"Education extraction completed in {edu_time:.3f}s")
        
        # Extract skills using SkillsExtractor
        skills_start = time.time()
        skills_extractor = SkillsExtractor()
        found_skills, category_flags, skills_confidence = skills_extractor.extract_skills(doc)
        result.is_tech = category_flags['is_tech']
        result.is_sales = category_flags['is_sales']
        result.is_marketing = category_flags['is_marketing']
        result.is_hr = category_flags['is_hr']
        result.confidence_scores['skills'] = skills_confidence
        result.extracted_entities['skills'] = found_skills
        skills_time = time.time() - skills_start
        logger.debug(f"Skills extraction completed in {skills_time:.3f}s")
        
        # Extract job level using JobLevelClassifier
        level_start = time.time()
        job_level_classifier = JobLevelClassifier()
        level_flags, level_confidence = job_level_classifier.classify_level(doc)
        result.is_junior = level_flags['is_junior']
        result.is_senior = level_flags['is_senior']
        result.is_manager = level_flags['is_manager']
        result.is_executive = level_flags['is_executive']
        result.confidence_scores['job_level'] = level_confidence
        level_time = time.time() - level_start
        logger.debug(f"Job level classification completed in {level_time:.3f}s")
        
        # For remaining fields (age, gender), use regex fallback
        regex_result = self._parse_with_regex(text)
        result.age = regex_result.age
        result.gender = regex_result.gender
        
        # Copy confidence scores for fields still using regex
        result.confidence_scores.update({
            'age': regex_result.confidence_scores.get('age', 0.5),
            'gender': regex_result.confidence_scores.get('gender', 0.6)
        })
        
        # Calculate overall confidence score (weighted average)
        overall_confidence = self._calculate_overall_confidence(result.confidence_scores)
        result.confidence_scores['overall'] = overall_confidence
        
        # Calculate total extraction time
        total_time = time.time() - start_time
        
        # Log comprehensive parsing summary
        logger.info(
            f"spaCy parsing completed successfully | "
            f"Method: spacy | "
            f"Overall confidence: {overall_confidence:.2f} | "
            f"Total time: {total_time:.3f}s | "
            f"Performance breakdown: NLP={nlp_time:.3f}s, "
            f"Experience={exp_time:.3f}s, Education={edu_time:.3f}s, "
            f"Skills={skills_time:.3f}s, JobLevel={level_time:.3f}s"
        )
        
        # Log detailed confidence scores
        logger.info(
            f"Confidence scores: "
            f"years_exp={exp_confidence:.2f}, "
            f"education={edu_confidence:.2f}, "
            f"skills={skills_confidence:.2f}, "
            f"job_level={level_confidence:.2f}, "
            f"overall={overall_confidence:.2f}"
        )
        
        # Log extracted values summary
        logger.info(
            f"Extracted values: "
            f"years_exp={years_exp}, "
            f"education_level={education_level}, "
            f"is_tech={result.is_tech}, "
            f"is_senior={result.is_senior}, "
            f"is_manager={result.is_manager}, "
            f"is_executive={result.is_executive}, "
            f"is_junior={result.is_junior}"
        )
        
        return result
    
    def _calculate_overall_confidence(self, confidence_scores: Dict[str, float]) -> float:
        """
        Calculate overall confidence score from individual extractor confidences.
        
        Uses weighted average based on importance of each field for salary prediction.
        
        Args:
            confidence_scores: Dictionary of confidence scores from extractors
            
        Returns:
            Overall confidence score between 0.0 and 1.0
        """
        # Define weights for each field (based on importance for salary prediction)
        weights = {
            'years_exp': 0.30,      # Most important factor
            'education_level': 0.25, # Very important
            'skills': 0.20,          # Important for tech roles
            'job_level': 0.15,       # Moderately important
            'age': 0.05,             # Less important
            'gender': 0.05           # Less important
        }
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for field, weight in weights.items():
            if field in confidence_scores:
                weighted_sum += confidence_scores[field] * weight
                total_weight += weight
        
        # Avoid division by zero
        if total_weight == 0:
            return 0.5
        
        overall = weighted_sum / total_weight
        
        logger.debug(f"Calculated overall confidence: {overall:.2f} from {len(confidence_scores)} scores")
        return overall
    
    def _parse_with_regex(self, text: str) -> ParsedResume:
        """
        Parse resume using regex patterns (fallback method).
        
        Preserves the existing regex-based parsing logic from app_professional.py
        for backward compatibility and fallback scenarios.
        
        Args:
            text: Resume text to parse
            
        Returns:
            ParsedResume: Structured resume data
        """
        text_lower = text.lower()
        result = ParsedResume()
        
        # Extract years of experience - improved patterns
        years_exp = 0
        current_year = datetime.now().year
        
        # Try explicit experience statements first
        exp_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
            r'experience[:\s]+(\d+)\s*(?:years?|yrs?)',
            r'total\s+(?:of\s+)?(\d+)\s*(?:years?|yrs?)',
        ]
        for pattern in exp_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                years_exp = max([int(m) for m in matches])
                logger.debug(f"Found explicit experience: {years_exp} years")
                break
        
        # If no explicit statement, calculate from date ranges
        if years_exp == 0:
            # Enhanced date pattern matching
            date_patterns = [
                r'(\d{4})\s*[-–—to]+\s*(?:(\d{4})|present|current|now|till\s+date|ongoing)',
                r'(\d{1,2})/(\d{4})\s*[-–—to]+\s*(?:\d{1,2}/(\d{4})|present|current|now)',
            ]
            
            all_ranges = []
            for pattern in date_patterns:
                dates = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in dates:
                    if len(match) >= 2:
                        # Extract start and end years
                        if '/' in pattern:  # MM/YYYY format
                            start_year = int(match[1]) if len(match) > 1 else 0
                            end_year = int(match[2]) if len(match) > 2 and match[2] else current_year
                        else:  # YYYY format
                            start_year = int(match[0])
                            end_year = int(match[1]) if match[1] else current_year
                        
                        if 1980 <= start_year <= end_year <= current_year + 1:
                            all_ranges.append((start_year, end_year))
                            logger.debug(f"Found date range: {start_year} - {end_year}")
            
            # Merge overlapping ranges and calculate total
            if all_ranges:
                sorted_ranges = sorted(all_ranges)
                merged_ranges = []
                current_start, current_end = sorted_ranges[0]
                
                for start, end in sorted_ranges[1:]:
                    if start <= current_end + 1:  # Allow 1 year gap
                        current_end = max(current_end, end)
                    else:
                        merged_ranges.append((current_start, current_end))
                        current_start, current_end = start, end
                
                merged_ranges.append((current_start, current_end))
                years_exp = sum(end - start for start, end in merged_ranges)
                logger.debug(f"Calculated {years_exp} years from {len(merged_ranges)} merged ranges")
        
        result.years_exp = years_exp
        
        # Extract education
        education_keywords = {
            'phd': 5, 'doctorate': 5, 'ph.d': 5,
            'master': 4, 'mba': 4, 'm.tech': 4, 'm.s': 4,
            'bachelor': 3, 'b.tech': 3, 'b.e': 3, 'b.s': 3,
            'diploma': 2,
            'high school': 1
        }
        education_level = 3
        for keyword, level in education_keywords.items():
            if keyword in text_lower:
                education_level = max(education_level, level)
        
        result.education_level = education_level
        
        # Extract age
        age_pattern = r'(?:age|born)[:\s]+(\d{2})'
        age_matches = re.findall(age_pattern, text_lower)
        result.age = int(age_matches[0]) if age_matches else 30
        
        # Detect gender
        result.gender = int(any(word in text_lower for word in ['he/him', 'male', 'mr.', 'mr ']))
        
        # Job level
        result.is_senior = int(any(word in text_lower for word in ['senior', 'sr.', 'lead', 'principal']))
        result.is_manager = int(any(word in text_lower for word in ['manager', 'director', 'head', 'vp']))
        result.is_executive = int(any(word in text_lower for word in ['ceo', 'cto', 'cfo', 'president', 'executive']))
        result.is_junior = int(any(word in text_lower for word in ['junior', 'jr.', 'entry', 'intern']))
        result.is_mid_level = int(any(word in text_lower for word in ['mid-level', 'mid level', 'intermediate']))
        
        # Job category
        result.is_tech = int(any(word in text_lower for word in ['software', 'developer', 'engineer', 'data', 'analyst', 'scientist', 'programming']))
        result.is_sales = int(any(word in text_lower for word in ['sales', 'account']))
        result.is_marketing = int('marketing' in text_lower)
        result.is_hr = int(any(word in text_lower for word in ['hr', 'human resources', 'recruiter']))
        
        # Set default confidence scores for regex parsing
        result.confidence_scores = {
            'years_exp': 0.7 if years_exp > 0 else 0.3,
            'education_level': 0.8,
            'age': 0.5,
            'gender': 0.6,
            'job_level': 0.7,
            'job_category': 0.7
        }
        
        return result
