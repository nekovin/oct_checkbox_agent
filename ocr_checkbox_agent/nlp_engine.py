"""Natural Language Understanding Module for intelligent form comprehension."""

import re
import string
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from loguru import logger

try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from fuzzywuzzy import fuzz, process
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
except ImportError as e:
    logger.warning(f"NLP dependencies not fully available: {e}")
    logger.info("Install with: pip install spacy nltk scikit-learn fuzzywuzzy")


class QuestionType(Enum):
    """Types of questions that can be identified."""
    RATING_SCALE = "rating_scale"
    MULTIPLE_CHOICE = "multiple_choice"
    YES_NO = "yes_no"
    TEXT_INPUT = "text_input"
    NUMERIC = "numeric"
    DATE = "date"
    LIKERT_SCALE = "likert_scale"
    RANKING = "ranking"
    CHECKLIST = "checklist"
    UNKNOWN = "unknown"


class ResponseType(Enum):
    """Types of responses expected."""
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    TEXT_BOX = "text_box"
    DROPDOWN = "dropdown"
    SLIDER = "slider"
    SMILEY_FACE = "smiley_face"
    STAR_RATING = "star_rating"
    NUMBER_SCALE = "number_scale"
    SIGNATURE = "signature"
    UNKNOWN = "unknown"


@dataclass
class QuestionEntity:
    """Represents a parsed question with metadata."""
    text: str
    question_type: QuestionType
    response_type: ResponseType
    options: List[str] = field(default_factory=list)
    scale_range: Optional[Tuple[int, int]] = None
    is_required: bool = False
    sub_questions: List['QuestionEntity'] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormSection:
    """Represents a section or group of related questions."""
    title: str
    questions: List[QuestionEntity]
    bbox: Optional[Tuple[int, int, int, int]] = None
    section_type: str = "general"
    instructions: str = ""


class NLPEngine:
    """Natural Language Understanding engine for form comprehension."""
    
    def __init__(self):
        """Initialize NLP engine with language models and patterns."""
        self.nlp = None
        self.lemmatizer = None
        self.stop_words = set()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Question patterns and indicators
        self._setup_patterns()
        
        logger.info("NLP Engine initialized")
    
    def _initialize_nlp(self):
        """Initialize spaCy and NLTK components."""
        try:
            # Try to load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Initialize NLTK components
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.info("Downloading required NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP components: {e}")
    
    def _setup_patterns(self):
        """Setup patterns for question and response type detection."""
        
        # Question type patterns
        self.question_patterns = {
            QuestionType.RATING_SCALE: [
                r'rate.*\b(1-[0-9]|scale|out of)',
                r'how would you rate',
                r'on a scale',
                r'from \d+ to \d+',
                r'satisfaction.*level'
            ],
            QuestionType.LIKERT_SCALE: [
                r'strongly (agree|disagree)',
                r'agree.*disagree',
                r'satisfaction.*level',
                r'(very|extremely).*satisfied',
                r'(excellent|good|fair|poor)'
            ],
            QuestionType.YES_NO: [
                r'\b(yes|no)\b.*\?',
                r'(agree|disagree)\s*\?',
                r'would you.*\?',
                r'do you.*\?',
                r'have you.*\?',
                r'are you.*\?'
            ],
            QuestionType.MULTIPLE_CHOICE: [
                r'choose.*from',
                r'select.*option',
                r'which of the following',
                r'pick.*that apply',
                r'check all'
            ],
            QuestionType.TEXT_INPUT: [
                r'please (explain|describe|write)',
                r'comments?',
                r'additional.*feedback',
                r'tell us about',
                r'in your own words'
            ],
            QuestionType.NUMERIC: [
                r'how many',
                r'number of',
                r'age',
                r'quantity',
                r'amount'
            ],
            QuestionType.DATE: [
                r'date.*birth',
                r'when.*born',
                r'mm/dd/yyyy',
                r'dd/mm/yyyy'
            ]
        }
        
        # Response type indicators
        self.response_indicators = {
            ResponseType.CHECKBOX: [
                'check all that apply',
                'select multiple',
                'mark all',
                '☐', '☑', '✓', '□', '■'
            ],
            ResponseType.RADIO_BUTTON: [
                'select one',
                'choose one',
                'pick one',
                '○', '●', '◯', '⚫'
            ],
            ResponseType.SMILEY_FACE: [
                '☺', '😊', '😐', '😞', '😢',
                'smiley', 'emoji', 'face'
            ],
            ResponseType.STAR_RATING: [
                '★', '☆', '⭐',
                'star rating', 'stars'
            ],
            ResponseType.TEXT_BOX: [
                'write in',
                'text box',
                'please explain',
                'comments'
            ]
        }
        
        # Scale indicators
        self.scale_patterns = [
            r'1\s*[-–]\s*([0-9]+)',
            r'([0-9]+)\s*[-–]\s*([0-9]+)',
            r'scale.*?([0-9]+).*?([0-9]+)',
            r'from\s+([0-9]+)\s+to\s+([0-9]+)'
        ]
        
        # Common question words and phrases
        self.question_starters = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which', 'would', 'do', 'did',
            'are', 'is', 'was', 'were', 'have', 'has', 'had', 'can', 'could', 'should',
            'please', 'rate', 'select', 'choose', 'indicate', 'mark', 'check'
        }
    
    def extract_questions(self, text_blocks: List[Dict[str, Any]]) -> List[QuestionEntity]:
        """Extract and classify questions from text blocks."""
        questions = []
        
        # Group text blocks into potential questions
        question_candidates = self._identify_question_candidates(text_blocks)
        
        for candidate in question_candidates:
            question = self._parse_question(candidate)
            if question:
                questions.append(question)
        
        # Post-process questions for relationships and context
        questions = self._enhance_question_context(questions)
        
        logger.info(f"Extracted {len(questions)} questions from {len(text_blocks)} text blocks")
        return questions
    
    def _identify_question_candidates(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential question text from blocks."""
        candidates = []
        
        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue
            
            # Check if text looks like a question
            if self._is_question_like(text):
                candidates.append(block)
            
            # Check for question numbers or bullets
            elif re.match(r'^(\d+\.|\d+\)|\*|-|•)', text):
                candidates.append(block)
            
            # Check for question indicators
            elif any(starter in text.lower() for starter in self.question_starters):
                candidates.append(block)
        
        return candidates
    
    def _is_question_like(self, text: str) -> bool:
        """Determine if text is likely a question."""
        text_lower = text.lower().strip()
        
        # Ends with question mark
        if text_lower.endswith('?'):
            return True
        
        # Starts with question words
        first_words = text_lower.split()[:3]
        if any(word in self.question_starters for word in first_words):
            return True
        
        # Contains imperative phrases
        imperative_phrases = ['please rate', 'select', 'choose', 'indicate', 'mark']
        if any(phrase in text_lower for phrase in imperative_phrases):
            return True
        
        # Has question pattern
        for patterns in self.question_patterns.values():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                return True
        
        return False
    
    def _parse_question(self, block: Dict[str, Any]) -> Optional[QuestionEntity]:
        """Parse a question block into a QuestionEntity."""
        text = block.get('text', '').strip()
        if not text:
            return None
        
        # Determine question type
        question_type = self._classify_question_type(text)
        
        # Determine response type
        response_type = self._classify_response_type(text)
        
        # Extract options if applicable
        options = self._extract_options(text, question_type)
        
        # Extract scale range
        scale_range = self._extract_scale_range(text)
        
        # Determine if required
        is_required = self._is_required_question(text)
        
        # Calculate confidence
        confidence = self._calculate_question_confidence(text, question_type, response_type)
        
        question = QuestionEntity(
            text=text,
            question_type=question_type,
            response_type=response_type,
            options=options,
            scale_range=scale_range,
            is_required=is_required,
            bbox=block.get('bbox'),
            confidence=confidence,
            context={'original_block': block}
        )
        
        return question
    
    def _classify_question_type(self, text: str) -> QuestionType:
        """Classify the type of question based on text analysis."""
        text_lower = text.lower()
        
        # Check patterns for each question type
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return q_type
        
        # Additional heuristics
        if '?' in text and any(word in text_lower for word in ['yes', 'no']):
            return QuestionType.YES_NO
        
        if any(word in text_lower for word in ['rate', 'rating', 'scale']):
            return QuestionType.RATING_SCALE
        
        if 'explain' in text_lower or 'describe' in text_lower:
            return QuestionType.TEXT_INPUT
        
        return QuestionType.UNKNOWN
    
    def _classify_response_type(self, text: str) -> ResponseType:
        """Classify the expected response type."""
        text_lower = text.lower()
        
        # Check for explicit indicators
        for r_type, indicators in self.response_indicators.items():
            for indicator in indicators:
                if indicator.lower() in text_lower:
                    return r_type
        
        # Infer from question type
        question_type = self._classify_question_type(text)
        
        if question_type == QuestionType.YES_NO:
            return ResponseType.RADIO_BUTTON
        elif question_type == QuestionType.RATING_SCALE:
            if 'smiley' in text_lower or 'face' in text_lower:
                return ResponseType.SMILEY_FACE
            elif 'star' in text_lower:
                return ResponseType.STAR_RATING
            else:
                return ResponseType.NUMBER_SCALE
        elif question_type == QuestionType.MULTIPLE_CHOICE:
            if 'all that apply' in text_lower:
                return ResponseType.CHECKBOX
            else:
                return ResponseType.RADIO_BUTTON
        elif question_type == QuestionType.TEXT_INPUT:
            return ResponseType.TEXT_BOX
        
        return ResponseType.UNKNOWN
    
    def _extract_options(self, text: str, question_type: QuestionType) -> List[str]:
        """Extract response options from question text."""
        options = []
        
        if question_type == QuestionType.YES_NO:
            return ['Yes', 'No']
        
        elif question_type == QuestionType.LIKERT_SCALE:
            # Standard Likert scale options
            if 'agree' in text.lower():
                return ['Strongly Agree', 'Agree', 'Neutral', 'Disagree', 'Strongly Disagree']
            elif 'satisfied' in text.lower():
                return ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']
        
        # Look for explicit options in parentheses or after colons
        option_patterns = [
            r'\((.*?)\)',
            r':\s*([A-Za-z][\w\s,/]+)',
            r'[A-Z]\.\s*([A-Za-z][\w\s]+)',
            r'[-•]\s*([A-Za-z][\w\s]+)'
        ]
        
        for pattern in option_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if ',' in match:
                    options.extend([opt.strip() for opt in match.split(',')])
                else:
                    options.append(match.strip())
        
        return options
    
    def _extract_scale_range(self, text: str) -> Optional[Tuple[int, int]]:
        """Extract scale range from question text."""
        for pattern in self.scale_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if len(match.groups()) == 1:
                        return (1, int(match.group(1)))
                    else:
                        return (int(match.group(1)), int(match.group(2)))
                except ValueError:
                    continue
        
        return None
    
    def _is_required_question(self, text: str) -> bool:
        """Determine if a question is required."""
        required_indicators = ['required', 'mandatory', 'must', '*', '(required)']
        return any(indicator in text.lower() for indicator in required_indicators)
    
    def _calculate_question_confidence(self, text: str, question_type: QuestionType, 
                                     response_type: ResponseType) -> float:
        """Calculate confidence score for question classification."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear question markers
        if text.endswith('?'):
            confidence += 0.2
        
        # Boost for question starters
        first_words = text.lower().split()[:2]
        if any(word in self.question_starters for word in first_words):
            confidence += 0.15
        
        # Boost for type-specific patterns
        if question_type != QuestionType.UNKNOWN:
            confidence += 0.2
        
        if response_type != ResponseType.UNKNOWN:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _enhance_question_context(self, questions: List[QuestionEntity]) -> List[QuestionEntity]:
        """Enhance questions with context and relationships."""
        for i, question in enumerate(questions):
            # Look for sub-questions (questions that follow and are related)
            for j in range(i + 1, min(i + 3, len(questions))):
                other_question = questions[j]
                
                # Check if it's a sub-question
                if self._is_sub_question(question.text, other_question.text):
                    question.sub_questions.append(other_question)
            
            # Enhance with semantic analysis if spaCy is available
            if self.nlp:
                question.context.update(self._analyze_semantic_context(question.text))
        
        return questions
    
    def _is_sub_question(self, main_text: str, candidate_text: str) -> bool:
        """Determine if candidate is a sub-question of main question."""
        # Simple heuristics for sub-questions
        main_lower = main_text.lower()
        candidate_lower = candidate_text.lower()
        
        # Check for numbering patterns
        main_starts_with_number = re.match(r'^\d+\.', main_text)
        candidate_starts_with_letter = re.match(r'^\s*[a-z][\).]', candidate_text)
        
        if main_starts_with_number and candidate_starts_with_letter:
            return True
        
        # Check for semantic similarity
        if self.nlp:
            try:
                main_doc = self.nlp(main_text)
                candidate_doc = self.nlp(candidate_text)
                similarity = main_doc.similarity(candidate_doc)
                return similarity > 0.7
            except:
                pass
        
        return False
    
    def _analyze_semantic_context(self, text: str) -> Dict[str, Any]:
        """Analyze semantic context using spaCy."""
        if not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            
            context = {
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'key_phrases': [chunk.text for chunk in doc.noun_chunks],
                'sentiment': 'neutral'  # Could integrate sentiment analysis
            }
            
            return context
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
            return {}
    
    def find_similar_questions(self, target_question: str, 
                             question_list: List[str], 
                             threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find questions similar to target using fuzzy matching."""
        if not question_list:
            return []
        
        # Use fuzzy matching to find similar questions
        matches = process.extract(target_question, question_list, 
                                scorer=fuzz.token_sort_ratio, limit=5)
        
        # Filter by threshold
        return [(match[0], match[1]/100.0) for match in matches if match[1]/100.0 >= threshold]
    
    def detect_form_sections(self, text_blocks: List[Dict[str, Any]]) -> List[FormSection]:
        """Detect form sections and group related questions."""
        sections = []
        current_section = None
        
        for block in text_blocks:
            text = block.get('text', '').strip()
            if not text:
                continue
            
            # Check if this is a section header
            if self._is_section_header(text):
                # Save previous section
                if current_section and current_section.questions:
                    sections.append(current_section)
                
                # Start new section
                current_section = FormSection(
                    title=text,
                    questions=[],
                    bbox=block.get('bbox'),
                    section_type=self._classify_section_type(text)
                )
            
            elif self._is_question_like(text):
                # Add question to current section
                if current_section is None:
                    current_section = FormSection(
                        title="General Questions",
                        questions=[],
                        section_type="general"
                    )
                
                question = self._parse_question(block)
                if question:
                    current_section.questions.append(question)
        
        # Add final section
        if current_section and current_section.questions:
            sections.append(current_section)
        
        return sections
    
    def _is_section_header(self, text: str) -> bool:
        """Determine if text is a section header."""
        # Headers are usually short, capitalized, and don't end with question marks
        if len(text) > 100 or text.endswith('?'):
            return False
        
        # Check for header patterns
        header_patterns = [
            r'^(SECTION|PART|CHAPTER)\s+[IVX\d]+',
            r'^[A-Z][A-Z\s]{5,30}$',  # All caps text
            r'^\d+\.\s*[A-Z][A-Za-z\s]+$'  # Numbered headers
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, text):
                return True
        
        # Check if mostly capitalized
        words = text.split()
        if len(words) > 1:
            cap_words = sum(1 for word in words if word[0].isupper())
            if cap_words / len(words) > 0.7:
                return True
        
        return False
    
    def _classify_section_type(self, title: str) -> str:
        """Classify the type of section based on title."""
        title_lower = title.lower()
        
        section_types = {
            'demographics': ['demographic', 'personal', 'background', 'about you'],
            'satisfaction': ['satisfaction', 'rating', 'feedback', 'evaluation'],
            'experience': ['experience', 'service', 'quality'],
            'contact': ['contact', 'information', 'details'],
            'preferences': ['preference', 'choice', 'selection']
        }
        
        for section_type, keywords in section_types.items():
            if any(keyword in title_lower for keyword in keywords):
                return section_type
        
        return "general"