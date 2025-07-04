"""Context Engine for intelligent question-response mapping and validation."""

import math
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .nlp_engine import QuestionEntity, QuestionType, ResponseType
from .form_analyzer import FormLayout, ResponseArea, FormSection
from .response_detector import ResponseValue, ResponseState


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    message: str
    question_id: Optional[str] = None
    response_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionResponsePair:
    """Represents a validated question-response pair."""
    question: QuestionEntity
    responses: List[ResponseValue]
    confidence: float
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormContext:
    """Complete context for a form including all validated pairs."""
    sections: List[FormSection]
    question_response_pairs: List[QuestionResponsePair]
    overall_confidence: float
    completion_rate: float
    validation_summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextEngine:
    """Intelligent context engine for form understanding and validation."""
    
    def __init__(self, config=None):
        """Initialize the context engine."""
        self.config = config
        
        # Validation rules
        self.validation_rules = self._setup_validation_rules()
        
        # Context analysis parameters
        self.proximity_threshold = 150  # pixels
        self.confidence_threshold = 0.6
        self.similarity_threshold = 0.8
        
        logger.info("Context Engine initialized")
    
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup validation rules for different question types."""
        
        return {
            'response_count': {
                QuestionType.YES_NO: {'min': 0, 'max': 1},
                QuestionType.RATING_SCALE: {'min': 0, 'max': 1},
                QuestionType.MULTIPLE_CHOICE: {'min': 0, 'max': 10},  # Variable
                QuestionType.TEXT_INPUT: {'min': 0, 'max': 1},
                QuestionType.NUMERIC: {'min': 0, 'max': 1}
            },
            'value_constraints': {
                QuestionType.RATING_SCALE: self._validate_rating_value,
                QuestionType.NUMERIC: self._validate_numeric_value,
                QuestionType.YES_NO: self._validate_yes_no_value
            },
            'consistency_checks': {
                'demographic_consistency': self._check_demographic_consistency,
                'logical_consistency': self._check_logical_consistency,
                'temporal_consistency': self._check_temporal_consistency
            }
        }
    
    def analyze_form_context(self, form_layout: FormLayout,
                           all_responses: List[ResponseValue],
                           image_metadata: Dict[str, Any]) -> FormContext:
        """Analyze complete form context and create validated pairs."""
        
        # Step 1: Map questions to responses
        question_response_pairs = self._map_questions_to_responses(
            form_layout, all_responses
        )
        
        # Step 2: Validate individual pairs
        for pair in question_response_pairs:
            self._validate_question_response_pair(pair)
        
        # Step 3: Cross-validate relationships
        self._cross_validate_responses(question_response_pairs)
        
        # Step 4: Calculate metrics
        overall_confidence = self._calculate_overall_confidence(question_response_pairs)
        completion_rate = self._calculate_completion_rate(question_response_pairs)
        
        # Step 5: Generate validation summary
        validation_summary = self._generate_validation_summary(question_response_pairs)
        
        form_context = FormContext(
            sections=form_layout.sections,
            question_response_pairs=question_response_pairs,
            overall_confidence=overall_confidence,
            completion_rate=completion_rate,
            validation_summary=validation_summary,
            metadata={
                'total_questions': len(question_response_pairs),
                'total_responses': len(all_responses),
                'image_metadata': image_metadata
            }
        )
        
        logger.info(f"Form context analyzed: {len(question_response_pairs)} Q-R pairs, "
                   f"confidence: {overall_confidence:.2f}, completion: {completion_rate:.2f}")
        
        return form_context
    
    def _map_questions_to_responses(self, form_layout: FormLayout,
                                   all_responses: List[ResponseValue]) -> List[QuestionResponsePair]:
        """Map questions to their corresponding responses using spatial and contextual analysis."""
        
        pairs = []
        used_responses = set()
        
        # Get all questions from sections
        all_questions = []
        for section in form_layout.sections:
            all_questions.extend(section.questions)
        
        for question in all_questions:
            # Find responses for this question
            question_responses = self._find_responses_for_question(
                question, all_responses, used_responses
            )
            
            # Calculate confidence for this mapping
            mapping_confidence = self._calculate_mapping_confidence(
                question, question_responses
            )
            
            pair = QuestionResponsePair(
                question=question,
                responses=question_responses,
                confidence=mapping_confidence,
                metadata={
                    'mapping_method': 'spatial_contextual',
                    'response_count': len(question_responses)
                }
            )
            
            pairs.append(pair)
            
            # Mark responses as used
            for response in question_responses:
                used_responses.add(id(response))
        
        return pairs
    
    def _find_responses_for_question(self, question: QuestionEntity,
                                   all_responses: List[ResponseValue],
                                   used_responses: Set[int]) -> List[ResponseValue]:
        """Find responses that belong to a specific question."""
        
        if not question.bbox:
            return []
        
        question_responses = []
        
        # Calculate question position
        q_center_x = question.bbox[0] + question.bbox[2] // 2
        q_center_y = question.bbox[1] + question.bbox[3] // 2
        q_bottom = question.bbox[1] + question.bbox[3]
        
        # Find candidate responses
        candidates = []
        
        for response in all_responses:
            if id(response) in used_responses or not response.bbox:
                continue
            
            r_center_x = response.bbox[0] + response.bbox[2] // 2
            r_center_y = response.bbox[1] + response.bbox[3] // 2
            
            # Calculate spatial relationship
            distance = math.sqrt((q_center_x - r_center_x)**2 + (q_center_y - r_center_y)**2)
            
            # Prefer responses to the right or below the question
            spatial_score = 1.0
            if r_center_y >= q_bottom:  # Below question
                spatial_score *= 1.2
            elif r_center_x > q_center_x:  # To the right
                spatial_score *= 1.1
            
            # Check response type compatibility
            type_score = self._calculate_type_compatibility(question, response)
            
            # Overall score
            score = spatial_score * type_score * (1 / (1 + distance / 100))
            
            if distance <= self.proximity_threshold and score > 0.3:
                candidates.append((response, score, distance))
        
        # Sort by score and select best candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Determine how many responses to select based on question type
        max_responses = self._get_max_responses_for_question(question)
        
        for response, score, distance in candidates[:max_responses]:
            if score > 0.5:  # Minimum score threshold
                question_responses.append(response)
        
        return question_responses
    
    def _calculate_type_compatibility(self, question: QuestionEntity, 
                                    response: ResponseValue) -> float:
        """Calculate compatibility score between question and response types."""
        
        compatibility_matrix = {
            (QuestionType.YES_NO, ResponseType.CHECKBOX): 1.0,
            (QuestionType.YES_NO, ResponseType.RADIO_BUTTON): 1.0,
            (QuestionType.RATING_SCALE, ResponseType.NUMBER_SCALE): 1.0,
            (QuestionType.RATING_SCALE, ResponseType.STAR_RATING): 1.0,
            (QuestionType.RATING_SCALE, ResponseType.SMILEY_FACE): 1.0,
            (QuestionType.LIKERT_SCALE, ResponseType.RADIO_BUTTON): 1.0,
            (QuestionType.LIKERT_SCALE, ResponseType.NUMBER_SCALE): 0.8,
            (QuestionType.MULTIPLE_CHOICE, ResponseType.CHECKBOX): 1.0,
            (QuestionType.MULTIPLE_CHOICE, ResponseType.RADIO_BUTTON): 0.9,
            (QuestionType.TEXT_INPUT, ResponseType.TEXT_BOX): 1.0,
            (QuestionType.NUMERIC, ResponseType.TEXT_BOX): 1.0,
            (QuestionType.NUMERIC, ResponseType.NUMBER_SCALE): 0.9,
        }
        
        key = (question.question_type, response.response_type)
        return compatibility_matrix.get(key, 0.3)  # Default compatibility
    
    def _get_max_responses_for_question(self, question: QuestionEntity) -> int:
        """Determine maximum expected responses for a question type."""
        
        max_responses = {
            QuestionType.YES_NO: 1,
            QuestionType.RATING_SCALE: 1,
            QuestionType.LIKERT_SCALE: 1,
            QuestionType.TEXT_INPUT: 1,
            QuestionType.NUMERIC: 1,
            QuestionType.DATE: 1,
            QuestionType.MULTIPLE_CHOICE: 5,  # Allow multiple for "check all that apply"
            QuestionType.CHECKLIST: 10
        }
        
        return max_responses.get(question.question_type, 3)
    
    def _calculate_mapping_confidence(self, question: QuestionEntity,
                                    responses: List[ResponseValue]) -> float:
        """Calculate confidence score for question-response mapping."""
        
        if not responses:
            return 0.0
        
        # Base confidence from question and response confidences
        question_conf = question.confidence
        avg_response_conf = sum(r.confidence for r in responses) / len(responses)
        
        base_confidence = (question_conf + avg_response_conf) / 2
        
        # Adjust based on number of responses
        expected_count = self._get_expected_response_count(question)
        actual_count = len(responses)
        
        count_factor = 1.0
        if expected_count > 0:
            count_factor = min(1.0, actual_count / expected_count)
        
        # Adjust based on response types
        type_factor = 1.0
        for response in responses:
            type_comp = self._calculate_type_compatibility(question, response)
            type_factor = min(type_factor, type_comp)
        
        final_confidence = base_confidence * count_factor * type_factor
        
        return min(1.0, max(0.0, final_confidence))
    
    def _get_expected_response_count(self, question: QuestionEntity) -> int:
        """Get expected number of responses for a question."""
        
        expected_counts = {
            QuestionType.YES_NO: 1,
            QuestionType.RATING_SCALE: 1,
            QuestionType.LIKERT_SCALE: 1,
            QuestionType.TEXT_INPUT: 1,
            QuestionType.NUMERIC: 1,
            QuestionType.MULTIPLE_CHOICE: 1,  # Minimum expected
            QuestionType.CHECKLIST: 2  # At least 2 for checklist
        }
        
        return expected_counts.get(question.question_type, 1)
    
    def _validate_question_response_pair(self, pair: QuestionResponsePair) -> None:
        """Validate a single question-response pair."""
        
        question = pair.question
        responses = pair.responses
        
        # Check response count
        self._validate_response_count(pair)
        
        # Check response values
        for response in responses:
            self._validate_response_value(question, response, pair)
        
        # Check consistency within responses
        self._validate_response_consistency(pair)
    
    def _validate_response_count(self, pair: QuestionResponsePair) -> None:
        """Validate the number of responses for a question."""
        
        question_type = pair.question.question_type
        response_count = len(pair.responses)
        
        rules = self.validation_rules['response_count'].get(question_type, {})
        min_count = rules.get('min', 0)
        max_count = rules.get('max', 10)
        
        if response_count < min_count:
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Too few responses: expected at least {min_count}, got {response_count}",
                context={'expected_min': min_count, 'actual': response_count}
            )
            pair.validation_issues.append(issue)
        
        elif response_count > max_count:
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Too many responses: expected at most {max_count}, got {response_count}",
                context={'expected_max': max_count, 'actual': response_count}
            )
            pair.validation_issues.append(issue)
    
    def _validate_response_value(self, question: QuestionEntity, 
                                response: ResponseValue,
                                pair: QuestionResponsePair) -> None:
        """Validate a specific response value."""
        
        question_type = question.question_type
        validator = self.validation_rules['value_constraints'].get(question_type)
        
        if validator:
            try:
                is_valid, message = validator(question, response)
                if not is_valid:
                    issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=message,
                        context={'value': response.value, 'question_type': question_type}
                    )
                    pair.validation_issues.append(issue)
            except Exception as e:
                logger.warning(f"Validation error: {e}")
    
    def _validate_response_consistency(self, pair: QuestionResponsePair) -> None:
        """Validate consistency within responses to a single question."""
        
        responses = pair.responses
        
        if len(responses) <= 1:
            return
        
        # For single-choice questions, only one response should be selected
        if pair.question.question_type in [QuestionType.YES_NO, QuestionType.RATING_SCALE]:
            selected_responses = [r for r in responses if r.state in [ResponseState.SELECTED, ResponseState.CHECKED]]
            
            if len(selected_responses) > 1:
                issue = ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Multiple responses selected for single-choice question",
                    context={'selected_count': len(selected_responses)}
                )
                pair.validation_issues.append(issue)
    
    def _validate_rating_value(self, question: QuestionEntity, response: ResponseValue) -> Tuple[bool, str]:
        """Validate rating scale values."""
        
        if question.scale_range:
            min_val, max_val = question.scale_range
            
            if isinstance(response.value, (int, float)):
                if min_val <= response.value <= max_val:
                    return True, ""
                else:
                    return False, f"Rating {response.value} outside valid range {min_val}-{max_val}"
        
        return True, ""
    
    def _validate_numeric_value(self, question: QuestionEntity, response: ResponseValue) -> Tuple[bool, str]:
        """Validate numeric values."""
        
        if isinstance(response.value, str):
            try:
                float(response.value)
                return True, ""
            except ValueError:
                return False, f"Non-numeric value: {response.value}"
        
        return True, ""
    
    def _validate_yes_no_value(self, question: QuestionEntity, response: ResponseValue) -> Tuple[bool, str]:
        """Validate yes/no values."""
        
        valid_values = [True, False, 'yes', 'no', 'y', 'n', 1, 0]
        
        if response.value in valid_values:
            return True, ""
        else:
            return False, f"Invalid yes/no value: {response.value}"
    
    def _cross_validate_responses(self, pairs: List[QuestionResponsePair]) -> None:
        """Perform cross-validation between different question-response pairs."""
        
        # Run consistency checks
        for check_name, check_func in self.validation_rules['consistency_checks'].items():
            try:
                issues = check_func(pairs)
                
                # Add issues to relevant pairs
                for issue in issues:
                    if issue.question_id:
                        for pair in pairs:
                            if str(id(pair.question)) == issue.question_id:
                                pair.validation_issues.append(issue)
                                break
            except Exception as e:
                logger.warning(f"Cross-validation check {check_name} failed: {e}")
    
    def _check_demographic_consistency(self, pairs: List[QuestionResponsePair]) -> List[ValidationIssue]:
        """Check consistency in demographic responses."""
        issues = []
        
        # Example: Age and birth year consistency
        age_pair = None
        birth_year_pair = None
        
        for pair in pairs:
            question_text = pair.question.text.lower()
            if 'age' in question_text and pair.responses:
                age_pair = pair
            elif 'birth' in question_text and 'year' in question_text and pair.responses:
                birth_year_pair = pair
        
        if age_pair and birth_year_pair:
            try:
                age = int(age_pair.responses[0].value)
                birth_year = int(birth_year_pair.responses[0].value)
                current_year = 2024  # This should be dynamic
                
                expected_age = current_year - birth_year
                
                if abs(age - expected_age) > 1:  # Allow 1 year tolerance
                    issue = ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Age ({age}) inconsistent with birth year ({birth_year})",
                        context={'age': age, 'birth_year': birth_year, 'expected_age': expected_age}
                    )
                    issues.append(issue)
            except (ValueError, IndexError):
                pass
        
        return issues
    
    def _check_logical_consistency(self, pairs: List[QuestionResponsePair]) -> List[ValidationIssue]:
        """Check logical consistency between related questions."""
        issues = []
        
        # Example: If someone answers "No" to "Do you own a car?", 
        # they shouldn't answer questions about car usage
        
        return issues
    
    def _check_temporal_consistency(self, pairs: List[QuestionResponsePair]) -> List[ValidationIssue]:
        """Check temporal consistency in date-related responses."""
        issues = []
        
        # Example: Start date should be before end date
        
        return issues
    
    def _calculate_overall_confidence(self, pairs: List[QuestionResponsePair]) -> float:
        """Calculate overall confidence for the form."""
        
        if not pairs:
            return 0.0
        
        confidences = [pair.confidence for pair in pairs]
        
        # Weight by number of validation issues
        weighted_confidences = []
        for pair in pairs:
            weight = 1.0
            
            # Reduce weight for pairs with issues
            error_count = sum(1 for issue in pair.validation_issues 
                            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
            
            if error_count > 0:
                weight *= 0.5 ** error_count
            
            weighted_confidences.append(pair.confidence * weight)
        
        return sum(weighted_confidences) / len(weighted_confidences)
    
    def _calculate_completion_rate(self, pairs: List[QuestionResponsePair]) -> float:
        """Calculate completion rate for the form."""
        
        if not pairs:
            return 0.0
        
        completed_pairs = 0
        
        for pair in pairs:
            if pair.responses:
                # Check if any response is actually filled/selected
                has_meaningful_response = any(
                    response.state in [ResponseState.CHECKED, ResponseState.SELECTED, 
                                     ResponseState.FILLED] and response.value is not None
                    for response in pair.responses
                )
                
                if has_meaningful_response:
                    completed_pairs += 1
        
        return completed_pairs / len(pairs)
    
    def _generate_validation_summary(self, pairs: List[QuestionResponsePair]) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        
        summary = {
            'total_questions': len(pairs),
            'completed_questions': 0,
            'questions_with_issues': 0,
            'issue_counts': {
                'info': 0,
                'warning': 0,
                'error': 0,
                'critical': 0
            },
            'common_issues': [],
            'quality_score': 0.0
        }
        
        issue_messages = []
        
        for pair in pairs:
            if pair.responses:
                summary['completed_questions'] += 1
            
            if pair.validation_issues:
                summary['questions_with_issues'] += 1
                
                for issue in pair.validation_issues:
                    summary['issue_counts'][issue.severity.value] += 1
                    issue_messages.append(issue.message)
        
        # Find common issues
        if issue_messages:
            from collections import Counter
            common = Counter(issue_messages).most_common(3)
            summary['common_issues'] = [{'message': msg, 'count': count} for msg, count in common]
        
        # Calculate quality score
        total_issues = sum(summary['issue_counts'].values())
        error_weight = summary['issue_counts']['error'] * 2 + summary['issue_counts']['critical'] * 4
        
        if len(pairs) > 0:
            quality_score = max(0, 1.0 - (error_weight / len(pairs)))
        else:
            quality_score = 0.0
        
        summary['quality_score'] = quality_score
        
        return summary