#services/metadata.py for metadata extraction and document type detection
import json
import spacy
from dateutil import parser as date_parser
from rapidfuzz import fuzz, process
import pandas as pd
import numpy as np
from typing import Pattern
import hashlib
import logging
import re
import os
import threading
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import torch
from transformers import (
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datetime import datetime
from collections import defaultdict, Counter
import time
from functools import lru_cache
from django.utils import timezone
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Philippines timezone
PH_TZ = pytz.timezone('Asia/Manila')

def get_ph_time():
    """Get current Philippines time"""
    return timezone.now().astimezone(PH_TZ)

def format_ph_time(dt=None):
    """Format Philippines time for logging"""
    if dt is None:
        dt = get_ph_time()
    return dt.strftime('%Y-%m-%d %H:%M:%S')

################################################################################
# Configuration for DocumentProcessor Compatibility
################################################################################

@dataclass
class ExtractionConfig:
    """Configuration for metadata extraction - compatible with DocumentProcessor"""
    # Model settings
    model_size: str = "7B"
    device: str = "auto"
    use_quantization: bool = True
    max_sequence_length: int = 2048
    
    # Processing settings - optimized for single file processing
    ai_timeout_seconds: int = 30
    enable_caching: bool = True
    cache_size: int = 500
    
    # Quality settings
    min_confidence_threshold: float = 0.3
    enable_validation: bool = True
    fallback_to_patterns: bool = True
    
    # Output settings
    include_debug_info: bool = False
    max_entities_per_type: int = 10
    
    # Performance settings for DocumentProcessor compatibility
    fast_mode: bool = True  # Skip heavy AI processing for speed
    pattern_only_mode: bool = False  # Use only pattern extraction

################################################################################
# Enhanced Pattern Matcher with Caching
################################################################################

class OptimizedPatternMatcher:
    """High-performance pattern matcher optimized for DocumentProcessor"""
    
    def __init__(self, config: ExtractionConfig):
        init_start_time = get_ph_time()
        self.config = config
        self._compiled_patterns = {}
        self._match_cache = {} if config.enable_caching else None
        self._cache_lock = threading.Lock()
        
        # Pre-compile common patterns for speed
        self._precompile_patterns()
        
        init_complete_time = get_ph_time()
        init_duration = (init_complete_time - init_start_time).total_seconds()
        logger.debug(f"[{format_ph_time(init_complete_time)}] OptimizedPatternMatcher initialized in {init_duration:.3f}s")
    
    def _precompile_patterns(self):
        """Pre-compile frequently used patterns"""
        compile_start_time = get_ph_time()
        
        common_patterns = {
            'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'amount_php': r'₱\s*([\d,]+\.?\d*)',
            'amount_usd': r'\$\s*([\d,]+\.?\d*)',
            'date_mdy': r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            'invoice_no': r'invoice\s+(?:no\.?|number)\s*:?\s*([A-Z0-9-]+)',
            'person_name': r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]+)*)',
        }
        
        for name, pattern in common_patterns.items():
            self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        
        compile_complete_time = get_ph_time()
        compile_duration = (compile_complete_time - compile_start_time).total_seconds()
        logger.debug(f"[{format_ph_time(compile_complete_time)}] Pre-compiled {len(common_patterns)} patterns in {compile_duration:.3f}s")
    
    @lru_cache(maxsize=1000)
    def compile_pattern(self, pattern: str) -> re.Pattern:
        """Compile and cache regex patterns"""
        return re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    
    def find_matches_fast(self, text: str, pattern_name: str) -> List[Dict]:
        """Fast pattern matching using pre-compiled patterns"""
        if pattern_name in self._compiled_patterns:
            pattern = self._compiled_patterns[pattern_name]
            matches = []
            for match in pattern.finditer(text):
                matches.append({
                    'value': match.group(1).strip() if match.groups() else match.group(0).strip(),
                    'pattern': pattern_name,
                    'position': match.start(),
                    'confidence': 0.8
                })
            return matches
        return []

################################################################################
# Fast Information Extractor - Optimized for DocumentProcessor
################################################################################

class FastInfoExtractor:
    """Fast information extraction optimized for DocumentProcessor workflow"""
    
    def __init__(self, config: ExtractionConfig):
        extractor_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(extractor_start_time)}] Initializing FastInfoExtractor...")
        
        self.config = config
        self.pattern_matcher = OptimizedPatternMatcher(config)
        
        # Document type signatures for fast detection
        self.doc_signatures = {
            'email': ['from:', 'to:', 'sent:', 'subject:', '@', 'dear', 'regards'],
            'check': ['check', 'cheque', 'pay to the order of', 'account number'],
            'voucher': ['voucher', 'payment voucher', 'accounts payable', 'ap'],
            'request_for_payment': ['request for payment', 'payment request', 'reimbursement'],
            'electronic_ticket': ['itinerary', 'booking', 'e-ticket', 'flight', 'passenger'],
            'official_receipt': ['official receipt', 'or number', 'received from', 'bir form'],
            'invoice': ['invoice number', 'bill to', 'amount due', 'payment terms']
        }
        
        # Fast extraction patterns
        self.fast_patterns = {
            'amounts': [
                r'₱\s*([\d,]+\.?\d*)',  # PHP
                r'\$\s*([\d,]+\.?\d*)',  # USD
                r'(?:total|amount|gross|net)[\s:]*([0-9,]+\.?[0-9]*)',
                r'([0-9,]+\.[0-9]{2})\s*(?=\s*\||\s*$)',  # Table amounts
            ],
            'dates': [
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})',
                r'(\d{6})',  # YYMMDD format
            ],
            'people': [
                r'From:\s*([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]+)*)\s*<',
                r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*[A-Z][a-z]+)*)\s*<[^>]+@[^>]+>',
                r'Dear\s+(?:Mr\.?|Ms\.?|Mrs\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            ],
            'companies': [
                r'([A-Z][A-Z\s&]+(?:CORP|CORPORATION|INC|COMPANY|CO\.?|GROUP))',
                r'@([a-zA-Z0-9-]+\.[a-zA-Z]{2,})',
            ],
            'invoice_numbers': [
                r'invoice\s+(?:no\.?|number)\s*:?\s*([A-Z0-9-]+)',
                r'inv\.?\s*(?:no\.?|#)?\s*([A-Z0-9-]+)',
            ]
        }
        
        # Try to load spaCy for advanced processing (optional)
        self.nlp = None
        self._try_load_spacy()
        
        extractor_complete_time = get_ph_time()
        extractor_duration = (extractor_complete_time - extractor_start_time).total_seconds()
        logger.info(f"[{format_ph_time(extractor_complete_time)}] FastInfoExtractor initialized in {extractor_duration:.2f}s")
    
    def _try_load_spacy(self):
        """Try to load spaCy model for advanced NLP"""
        try:
            if not self.config.fast_mode:  # Only load if not in fast mode
                spacy_start_time = get_ph_time()
                self.nlp = spacy.load("en_core_web_sm")
                spacy_complete_time = get_ph_time()
                spacy_duration = (spacy_complete_time - spacy_start_time).total_seconds()
                logger.info(f"[{format_ph_time(spacy_complete_time)}] spaCy model loaded in {spacy_duration:.2f}s")
        except Exception as e:
            error_time = get_ph_time()
            logger.debug(f"[{format_ph_time(error_time)}] spaCy model not available: {e}")
            self.nlp = None
    
    def detect_document_type_fast(self, text: str, filename: str = "") -> Tuple[str, float]:
        """Fast document type detection"""
        detect_start_time = get_ph_time()
        
        if not text and not filename:
            return 'general', 0.3
        
        text_lower = text.lower() if text else ""
        filename_lower = filename.lower() if filename else ""
        
        scores = {}
        
        for doc_type, keywords in self.doc_signatures.items():
            score = 0
            
            # Check filename (high weight for speed)
            if filename_lower:
                filename_matches = sum(1 for kw in keywords if kw in filename_lower)
                score += filename_matches * 0.5
            
            # Check content (limited for speed)
            if text_lower:
                content_matches = sum(1 for kw in keywords[:3] if kw in text_lower)  # Only check first 3 keywords
                score += (content_matches / 3) * 0.5
            
            if score > 0:
                scores[doc_type] = score
        
        if scores:
            best_type = max(scores.keys(), key=lambda k: scores[k])
            confidence = min(scores[best_type], 1.0)
            
            # Special fast detection for emails
            if any(indicator in text_lower for indicator in ['from:', 'to:', 'subject:']):
                detect_complete_time = get_ph_time()
                logger.debug(f"[{format_ph_time(detect_complete_time)}] Fast email detection")
                return 'email', 0.9
            
            if confidence >= 0.2:
                detect_complete_time = get_ph_time()
                logger.debug(f"[{format_ph_time(detect_complete_time)}] Detected: {best_type} ({confidence:.2f})")
                return best_type, confidence
        
        return 'general', 0.3
    
    def extract_fast_metadata(self, text: str, doc_type: str = None, filename: str = "") -> dict:
        """Fast metadata extraction optimized for DocumentProcessor"""
        extraction_start_time = get_ph_time()
        logger.debug(f"[{format_ph_time(extraction_start_time)}] Starting fast metadata extraction...")
        
        # Quick validation
        if not text or len(text.strip()) < 10:
            fallback_result = self._extract_from_filename_only(filename, doc_type or 'general')
            extraction_complete_time = get_ph_time()
            logger.debug(f"[{format_ph_time(extraction_complete_time)}] Used filename-only extraction")
            return fallback_result
        
        # Document type detection
        if not doc_type or doc_type == 'general':
            detected_type, _ = self.detect_document_type_fast(text, filename)
        else:
            detected_type = doc_type
        
        # Fast extraction
        result = {
            "doc_type": detected_type,
            "title": self._extract_title_fast(text, detected_type, filename),
            "ids": self._extract_ids_fast(text, filename, detected_type),
            "parties": self._extract_parties_fast(text, detected_type),
            "dates": self._extract_dates_fast(text, filename),
            "amounts": self._extract_amounts_fast(text, detected_type),
            "line_items": [],
            "payment_terms": None,
            "addresses": {},
            "contacts": self._extract_contacts_fast(text),
            "status": "processed",
            "tags": [detected_type],
            "extracted_entities": self._extract_entities_fast(text, detected_type),
            "notes": [],
            "confidence": 0.0
        }
        
        # Fast confidence calculation
        result["confidence"] = self._calculate_fast_confidence(result, text, filename)
        
        extraction_complete_time = get_ph_time()
        extraction_duration = (extraction_complete_time - extraction_start_time).total_seconds()
        logger.debug(f"[{format_ph_time(extraction_complete_time)}] Fast metadata extraction completed in {extraction_duration:.3f}s")
        
        return result
    
    def _extract_title_fast(self, text: str, doc_type: str, filename: str) -> str:
        """Fast title extraction"""
        if doc_type == 'email':
            # Try to extract subject line
            subject_match = re.search(r'subject:\s*([^\n]+)', text, re.IGNORECASE)
            if subject_match:
                subject = subject_match.group(1).strip()
                if subject and len(subject) > 3:
                    return f"Email - {subject}"
            return "Email Communication"
        
        # For other documents, try to extract meaningful title from first few lines
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if 15 <= len(line) <= 80 and not re.match(r'^\d', line):
                return line
        
        # Fallback to filename-based title
        if filename:
            name_part = filename.replace('.pdf', '').replace('.PDF', '')
            return name_part if name_part else f"{doc_type.replace('_', ' ').title()}"
        
        return f"{doc_type.replace('_', ' ').title()}"
    
    def _extract_ids_fast(self, text: str, filename: str, doc_type: str) -> dict:
        """Fast ID extraction"""
        ids = {}
        
        # Fast pattern matching for common IDs
        if doc_type == 'invoice':
            patterns = self.fast_patterns['invoice_numbers']
        else:
            patterns = [
                r'(?:doc|ref|no)\.?\s*:?\s*([A-Z0-9-]{4,})',
                r'check\s*(?:no\.?)?\s*:?\s*([0-9]+)',
                r'voucher\s*(?:no\.?)?\s*:?\s*([A-Z0-9-]+)',
            ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                id_value = match.group(1).strip()
                if len(id_value) >= 3:
                    field_name = 'document_no' if doc_type == 'general' else f'{doc_type}_no'
                    ids[field_name] = id_value
                    break
        
        # Extract from filename if no ID found in text
        if not ids and filename:
            filename_nums = re.findall(r'\b([0-9]{6,})\b', filename)
            if filename_nums:
                ids['reference_no'] = filename_nums[0]
        
        return ids
    
    def _extract_parties_fast(self, text: str, doc_type: str) -> dict:
        """Fast party extraction"""
        parties = {
            "issuer": {"name": None, "tin": None},
            "recipient": {"name": None, "tin": None},
            "other": []
        }
        
        if doc_type == 'email':
            # Fast email party extraction
            from_match = re.search(r'from:\s*([^<\n]+?)(?:\s*<|$)', text, re.IGNORECASE)
            if from_match:
                from_name = from_match.group(1).strip()
                if self._is_valid_name_fast(from_name):
                    parties['issuer']['name'] = from_name
            
            to_match = re.search(r'to:\s*([^<\n]+?)(?:\s*<|$)', text, re.IGNORECASE)
            if to_match:
                to_name = to_match.group(1).strip()
                if self._is_valid_name_fast(to_name):
                    parties['recipient']['name'] = to_name
        else:
            # Fast general party extraction
            person_matches = self.pattern_matcher.find_matches_fast(text, 'person_name')
            if person_matches:
                parties['recipient']['name'] = person_matches[0]['value']
            
            # Fast company extraction
            company_match = re.search(r'([A-Z]+\s+(?:CORP|INC|COMPANY))', text, re.IGNORECASE)
            if company_match:
                parties['issuer']['name'] = company_match.group(1).strip()
        
        return parties
    
    def _extract_dates_fast(self, text: str, filename: str = "") -> dict:
        """Fast date extraction"""
        dates = {}
        
        # Fast date pattern matching
        for pattern in self.fast_patterns['dates']:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1).strip()
                
                # Handle YYMMDD format
                if re.match(r'^\d{6}$', date_str):
                    try:
                        year = int(date_str[:2]) + 2000
                        month = int(date_str[2:4])
                        day = int(date_str[4:6])
                        if 1 <= month <= 12 and 1 <= day <= 31:
                            dates['issue_date'] = f"{day:02d}/{month:02d}/{year}"
                            break
                    except ValueError:
                        continue
                else:
                    dates['issue_date'] = date_str
                    break
        
        # Try filename date extraction if no date found
        if not dates and filename:
            filename_date = re.search(r'(\d{6})', filename)
            if filename_date:
                date_str = filename_date.group(1)
                try:
                    year = int(date_str[:2]) + 2000
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        dates['issue_date'] = f"{day:02d}/{month:02d}/{year}"
                except ValueError:
                    pass
        
        return dates
    
    def _extract_amounts_fast(self, text: str, doc_type: str) -> dict:
        """Fast amount extraction"""
        amounts = {"currency": "PHP", "total": None, "subtotal": None, "tax": None}
        
        found_amounts = []
        currency = "PHP"
        
        # Fast amount pattern matching
        for pattern in self.fast_patterns['amounts']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(',', '').strip()
                
                # Detect currency from context
                context = text[max(0, match.start()-20):match.end()+20].upper()
                if any(curr in context for curr in ['USD', '$']):
                    currency = "USD"
                
                try:
                    amount_value = float(amount_str)
                    if 0.01 <= amount_value <= 100000000:  # Reasonable range
                        found_amounts.append(amount_value)
                except ValueError:
                    continue
        
        # Set currency and amounts
        amounts['currency'] = currency
        if found_amounts:
            amounts['total'] = max(found_amounts)  # Largest amount as total
        
        return amounts
    
    def _extract_contacts_fast(self, text: str) -> dict:
        """Fast contact extraction"""
        contacts = {}
        
        # Fast email extraction
        email_matches = self.pattern_matcher.find_matches_fast(text, 'email')
        if email_matches:
            emails = [match['value'] for match in email_matches]
            contacts['emails'] = list(set(emails))  # Remove duplicates
            contacts['email'] = emails[0]  # Primary email
        
        return contacts
    
    def _extract_entities_fast(self, text: str, doc_type: str) -> dict:
        """Fast entity extraction"""
        entities = {
            "people": [],
            "companies": [],
            "locations": [],
            "projects": []
        }
        
        # Fast people extraction
        person_matches = self.pattern_matcher.find_matches_fast(text, 'person_name')
        for match in person_matches[:5]:  # Limit to first 5 for speed
            name = match['value']
            if self._is_valid_name_fast(name):
                entities["people"].append(name)
        
        # Fast company extraction
        company_patterns = [
            r'([A-Z]+\s+(?:CORP|CORPORATION|INC|COMPANY))',
            r'EEI\s+(?:CORP|CORPORATION)',
            r'COMFAC\s+(?:CORP|GROUP)',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["companies"].append(match.group(1).strip())
                break  # Take first match for speed
        
        return entities
    
    def _extract_from_filename_only(self, filename: str, doc_type: str) -> dict:
        """Fast fallback extraction from filename when text is insufficient"""
        filename_start_time = get_ph_time()
        
        result = {
            "doc_type": doc_type,
            "title": filename.replace('.pdf', '') if filename else 'Unknown Document',
            "ids": {},
            "parties": {"issuer": {"name": None}, "recipient": {"name": None}},
            "dates": {},
            "amounts": {"currency": "PHP", "total": None},
            "line_items": [],
            "payment_terms": None,
            "addresses": {},
            "contacts": {},
            "status": "processed",
            "tags": [doc_type],
            "extracted_entities": {"people": [], "companies": []},
            "notes": ["Extracted from filename - insufficient text"],
            "confidence": 0.4
        }
        
        if filename:
            # Extract date from filename
            date_match = re.search(r'(\d{6})', filename)
            if date_match:
                date_str = date_match.group(1)
                try:
                    year = int(date_str[:2]) + 2000
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        result['dates']['issue_date'] = f"{day:02d}/{month:02d}/{year}"
                except ValueError:
                    pass
            
            # Extract document number
            numbers = re.findall(r'\b(\d{7,})\b', filename)
            if numbers:
                result['ids']['document_no'] = numbers[0]
        
        filename_complete_time = get_ph_time()
        filename_duration = (filename_complete_time - filename_start_time).total_seconds()
        logger.debug(f"[{format_ph_time(filename_complete_time)}] Filename extraction completed in {filename_duration:.3f}s")
        
        return result
    
    def _is_valid_name_fast(self, name: str) -> bool:
        """Fast name validation"""
        if not name or len(name.strip()) < 2:
            return False
        
        name = name.strip()
        
        # Quick rejection filters
        if any(noise in name.lower() for noise in ['subject', '@', 'corporation', 'payment']):
            return False
        
        # Basic validation
        words = name.split()
        if not (1 <= len(words) <= 4):
            return False
        
        return all(len(word) >= 2 and word[0].isupper() for word in words)
    
    def _calculate_fast_confidence(self, result: dict, text: str, filename: str) -> float:
        """Fast confidence calculation for DocumentProcessor compatibility"""
        confidence = 0.4  # Base confidence
        
        # Quick bonuses based on extracted data
        if result.get('ids') and any(result['ids'].values()):
            confidence += 0.15
        
        if result.get('dates') and any(result['dates'].values()):
            confidence += 0.1
        
        if result.get('amounts', {}).get('total'):
            confidence += 0.15
        
        if result.get('parties', {}).get('issuer', {}).get('name') or result.get('parties', {}).get('recipient', {}).get('name'):
            confidence += 0.1
        
        if result.get('doc_type') != 'general':
            confidence += 0.05
        
        if text and len(text) > 200:
            confidence += 0.1
        
        if result.get('contacts', {}).get('email'):
            confidence += 0.05
        
        return min(confidence, 0.95)

################################################################################
# Optimized Qwen Model Manager - Compatible with DocumentProcessor
################################################################################

class OptimizedQwenModelManager:
    """Memory-optimized model manager compatible with DocumentProcessor patterns"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._initialized = True
        init_start_time = get_ph_time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        
        init_complete_time = get_ph_time()
        init_duration = (init_complete_time - init_start_time).total_seconds()
        logger.info(f"[{format_ph_time(init_complete_time)}] OptimizedQwenModelManager initialized on {self.device} in {init_duration:.3f}s")
    
    def get_qwen25_model(self, model_size="7B"):
        """Get optimized Qwen 2.5 model with better memory management"""
        model_key = f"qwen25_instruct_{model_size}"
        
        if model_key not in self.models:
            with self._lock:
                if model_key not in self.models:
                    try:
                        load_start_time = get_ph_time()
                        logger.info(f"[{format_ph_time(load_start_time)}] Loading Qwen2.5-{model_size}-Instruct (optimized)")
                        model_name = f"Qwen/Qwen2.5-{model_size}-Instruct"
                        
                        # Optimized quantization for single-file processing
                        quant = None
                        if self.device == "cuda":
                            quant = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=False,
                                bnb_4bit_quant_type="nf4",
                            )
                        
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_name, 
                            trust_remote_code=True,
                            use_fast=True
                        )
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="auto" if self.device == "cuda" else None,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                            quantization_config=quant,
                            trust_remote_code=True,
                            attn_implementation="eager",
                            low_cpu_mem_usage=True,
                        )
                        
                        if self.device == "cpu":
                            model = model.to("cpu")
                        
                        model.eval()
                        
                        # Set up padding
                        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is None:
                            tokenizer.pad_token_id = tokenizer.eos_token_id
                        
                        self.models[model_key] = (model, tokenizer)
                        
                        load_complete_time = get_ph_time()
                        load_duration = (load_complete_time - load_start_time).total_seconds()
                        logger.info(f"[{format_ph_time(load_complete_time)}] Qwen2.5-{model_size}-Instruct loaded successfully in {load_duration:.2f}s")
                        
                        # Clean up memory after loading
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        error_time = get_ph_time()
                        logger.error(f"[{format_ph_time(error_time)}] Failed to load Qwen2.5 model: {e}")
                        self.models[model_key] = (None, None)
        
        return self.models.get(model_key, (None, None))
    
    def cleanup(self):
        """Cleanup for DocumentProcessor compatibility"""
        cleanup_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(cleanup_start_time)}] Starting OptimizedQwenModelManager cleanup...")
        
        with self._lock:
            for key, value in self.models.items():
                if value and value != (None, None):
                    model, tokenizer = value
                    if model is not None:
                        del model
                    if tokenizer is not None:
                        del tokenizer
            self.models.clear()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            cleanup_complete_time = get_ph_time()
            cleanup_duration = (cleanup_complete_time - cleanup_start_time).total_seconds()
            logger.info(f"[{format_ph_time(cleanup_complete_time)}] OptimizedQwenModelManager cleanup completed in {cleanup_duration:.2f}s")

################################################################################
# Enhanced QwenExtractor - DocumentProcessor Compatible
################################################################################

class QwenExtractor:
    """Enhanced QwenExtractor optimized for DocumentProcessor integration"""

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        config: Optional[dict] = None,
        model_size: str = "7B",
        **kwargs
    ):
        extractor_init_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(extractor_init_start_time)}] Initializing QwenExtractor...")
        
        # Configuration setup
        self.config = ExtractionConfig(
            model_size=model_size,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            fast_mode=kwargs.get('fast_mode', True),
            pattern_only_mode=kwargs.get('pattern_only_mode', False)
        )
        
        self.device = self.config.device
        self.model_size = model_size
        
        # Initialize fast info extractor (always available)
        self.info_extractor = FastInfoExtractor(self.config)
        
        # Initialize AI components only if not in pattern-only mode
        if not self.config.pattern_only_mode:
            ai_init_start_time = get_ph_time()
            self.model_manager = OptimizedQwenModelManager()
            self.model, self.tokenizer = self.model_manager.get_qwen25_model(self.model_size)
            
            if self.model is None or self.tokenizer is None:
                ai_warn_time = get_ph_time()
                logger.warning(f"[{format_ph_time(ai_warn_time)}] AI model unavailable, falling back to fast pattern extraction")
                self.use_ai = False
            else:
                self.use_ai = True and not self.config.fast_mode
                
                if self.use_ai:
                    self.generation_config = GenerationConfig(
                        max_new_tokens=300,
                        do_sample=False,
                        temperature=0.1,
                        top_p=0.9,
                        repetition_penalty=1.03,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True,
                    )
                
                ai_init_complete_time = get_ph_time()
                ai_init_duration = (ai_init_complete_time - ai_init_start_time).total_seconds()
                logger.debug(f"[{format_ph_time(ai_init_complete_time)}] AI components initialized in {ai_init_duration:.2f}s")
        else:
            self.use_ai = False
            self.model = None
            self.tokenizer = None
        
        extractor_init_complete_time = get_ph_time()
        extractor_init_duration = (extractor_init_complete_time - extractor_init_start_time).total_seconds()
        logger.info(f"[{format_ph_time(extractor_init_complete_time)}] QwenExtractor initialized in {extractor_init_duration:.2f}s - Model: {self.model_size}, AI: {self.use_ai}, Fast: {self.config.fast_mode}")

    def extract_json(self, ocr_text: str, doc_type: str = None, filename: str = "") -> Dict[str, Any]:
        """Main extraction method compatible with DocumentProcessor"""
        try:
            extract_start_time = get_ph_time()
            
            # Always use fast pattern extraction for DocumentProcessor compatibility
            result = self.info_extractor.extract_fast_metadata(
                ocr_text or "", 
                doc_type, 
                filename
            )
            
            # Optional AI enhancement (only if enabled and worthwhile)
            if (self.use_ai and 
                not self.config.fast_mode and 
                len(ocr_text or "") > 200 and 
                result.get('confidence', 0) < 0.8):
                
                try:
                    ai_enhance_start_time = get_ph_time()
                    ai_enhancement = self._quick_ai_enhance(ocr_text, result.get('doc_type'))
                    if ai_enhancement:
                        result = self._merge_enhancements(result, ai_enhancement)
                        result['extraction_method'] = 'ai_enhanced'
                        ai_enhance_complete_time = get_ph_time()
                        ai_enhance_duration = (ai_enhance_complete_time - ai_enhance_start_time).total_seconds()
                        logger.debug(f"[{format_ph_time(ai_enhance_complete_time)}] AI enhancement completed in {ai_enhance_duration:.3f}s")
                    else:
                        result['extraction_method'] = 'fast_pattern'
                except Exception as e:
                    error_time = get_ph_time()
                    logger.warning(f"[{format_ph_time(error_time)}] AI enhancement failed: {e}")
                    result['extraction_method'] = 'fast_pattern_fallback'
            else:
                result['extraction_method'] = 'fast_pattern'
            
            # Add processing metadata for DocumentProcessor compatibility
            extract_complete_time = get_ph_time()
            processing_time = (extract_complete_time - extract_start_time).total_seconds()
            result['processing_time'] = processing_time
            result['model_used'] = f'Enhanced-{self.model_size}'
            
            logger.debug(f"[{format_ph_time(extract_complete_time)}] JSON extraction completed in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            error_time = get_ph_time()
            logger.error(f"[{format_ph_time(error_time)}] Enhanced extraction failed: {e}")
            return self._create_fallback_result(doc_type or "general", str(e))

    @torch.inference_mode()
    def _quick_ai_enhance(self, text: str, doc_type: str) -> dict:
        """Quick AI enhancement with timeout protection"""
        if not self.use_ai or not self.model:
            return {}
        
        ai_start_time = get_ph_time()
        
        # Build minimal prompt for speed
        prompt = self._build_quick_prompt(text, doc_type)
        
        try:
            # Quick tokenization with aggressive truncation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,  # Reduced for DocumentProcessor speed requirements
                padding=False
            ).to(self.device)
            
            # Quick generation with timeout simulation
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                output_ids = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )
            
            # Quick decode
            generated_tokens = output_ids[0][len(inputs.input_ids[0]):]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            ai_complete_time = get_ph_time()
            ai_duration = (ai_complete_time - ai_start_time).total_seconds()
            logger.debug(f"[{format_ph_time(ai_complete_time)}] AI inference completed in {ai_duration:.3f}s")
            
            return self._parse_ai_response(response)
            
        except Exception as e:
            error_time = get_ph_time()
            logger.warning(f"[{format_ph_time(error_time)}] AI enhancement failed: {e}")
            return {}
    
    def _build_quick_prompt(self, text: str, doc_type: str) -> str:
        """Build minimal prompt for speed"""
        return f"""Extract key information from this {doc_type} document:

{text[:1500]}

Return JSON with:
- title: document title
- issuer: issuing party
- recipient: receiving party  
- amount: main amount with currency
- date: primary date
- confidence: extraction confidence (0-1)

JSON:"""

    def _parse_ai_response(self, response: str) -> dict:
        """Parse AI response with fallback strategies"""
        try:
            # Try direct JSON parse
            response = response.strip()
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
            
            # Find JSON block
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            if json_match:
                return json.loads(json_match.group(0))
            
            # Extract individual fields
            extracted = {}
            field_patterns = [
                (r'"title":\s*"([^"]*)"', 'title'),
                (r'"issuer":\s*"([^"]*)"', 'issuer'),
                (r'"recipient":\s*"([^"]*)"', 'recipient'),
                (r'"amount":\s*"?([^",}]*)"?', 'amount'),
                (r'"date":\s*"([^"]*)"', 'date'),
            ]
            
            for pattern, field in field_patterns:
                match = re.search(pattern, response)
                if match:
                    value = match.group(1).strip()
                    if value and value != "null":
                        extracted[field] = value
            
            return extracted
            
        except Exception as e:
            error_time = get_ph_time()
            logger.warning(f"[{format_ph_time(error_time)}] AI response parsing failed: {e}")
            return {}

    def _merge_enhancements(self, base_result: dict, ai_data: dict) -> dict:
        """Merge AI enhancement with base result"""
        result = base_result.copy()
        
        # Merge improvements
        if ai_data.get('title') and len(ai_data['title']) > 5:
            result['title'] = ai_data['title']
        
        if ai_data.get('issuer'):
            if 'parties' not in result:
                result['parties'] = {"issuer": {}, "recipient": {}}
            result['parties']['issuer']['name'] = ai_data['issuer']
        
        if ai_data.get('recipient'):
            if 'parties' not in result:
                result['parties'] = {"issuer": {}, "recipient": {}}
            result['parties']['recipient']['name'] = ai_data['recipient']
        
        if ai_data.get('amount'):
            try:
                amount_str = str(ai_data['amount']).replace(',', '').replace(', ', '').replace('PHP', '').strip()
                amount_val = float(amount_str)
                if 0.01 <= amount_val <= 100000000:
                    if 'amounts' not in result:
                        result['amounts'] = {"currency": "PHP"}
                    result['amounts']['total'] = amount_val
            except (ValueError, TypeError):
                pass
        
        if ai_data.get('date'):
            if 'dates' not in result:
                result['dates'] = {}
            result['dates']['issue_date'] = ai_data['date']
        
        # Boost confidence for successful AI enhancement
        if len(ai_data) >= 2:
            result['confidence'] = min(result.get('confidence', 0.5) + 0.15, 0.9)
        
        return result

    def _create_fallback_result(self, doc_type: str, error_msg: str = "") -> dict:
        """Create fallback result for DocumentProcessor compatibility"""
        fallback_time = get_ph_time()
        logger.debug(f"[{format_ph_time(fallback_time)}] Creating fallback result for {doc_type}")
        
        return {
            "doc_type": doc_type,
            "title": None,
            "ids": {},
            "parties": {"issuer": {"name": None, "tin": None}, "recipient": {"name": None, "tin": None}},
            "dates": {},
            "amounts": {"currency": "PHP", "total": None, "subtotal": None, "tax": None},
            "line_items": [],
            "payment_terms": None,
            "addresses": {},
            "contacts": {},
            "status": "processed",
            "tags": [doc_type],
            "extracted_entities": {"people": [], "companies": [], "locations": [], "projects": []},
            "notes": [f"Extraction failed: {error_msg}"] if error_msg else [],
            "confidence": 0.3,
            "extraction_method": "fallback",
            "error": error_msg
        }

    def cleanup(self):
        """Cleanup for DocumentProcessor compatibility"""
        cleanup_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(cleanup_start_time)}] Starting QwenExtractor cleanup...")
        
        try:
            self.model = None
            self.tokenizer = None
            if hasattr(self, 'model_manager'):
                # Don't cleanup shared model manager to avoid affecting other instances
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            cleanup_complete_time = get_ph_time()
            cleanup_duration = (cleanup_complete_time - cleanup_start_time).total_seconds()
            logger.info(f"[{format_ph_time(cleanup_complete_time)}] QwenExtractor cleanup completed in {cleanup_duration:.2f}s")
        except Exception as e:
            error_time = get_ph_time()
            logger.warning(f"[{format_ph_time(error_time)}] QwenExtractor cleanup warning: {e}")

################################################################################
# DocumentProcessor Integration Functions
################################################################################

def cleanup_qwen_models():
    """Global cleanup for DocumentProcessor compatibility"""
    cleanup_start_time = get_ph_time()
    logger.info(f"[{format_ph_time(cleanup_start_time)}] Starting global Qwen models cleanup...")
    
    try:
        OptimizedQwenModelManager().cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        cleanup_complete_time = get_ph_time()
        cleanup_duration = (cleanup_complete_time - cleanup_start_time).total_seconds()
        logger.info(f"[{format_ph_time(cleanup_complete_time)}] Global Qwen models cleanup completed in {cleanup_duration:.2f}s")
    except Exception as e:
        error_time = get_ph_time()
        logger.warning(f"[{format_ph_time(error_time)}] Qwen models cleanup failed: {e}")

def create_qwen_extractor(model_size: str = "7B", fast_mode: bool = True, **kwargs) -> QwenExtractor:
    """Create QwenExtractor optimized for DocumentProcessor"""
    create_start_time = get_ph_time()
    extractor = QwenExtractor(
        model_size=model_size, 
        fast_mode=fast_mode,
        **kwargs
    )
    create_complete_time = get_ph_time()
    create_duration = (create_complete_time - create_start_time).total_seconds()
    logger.debug(f"[{format_ph_time(create_complete_time)}] QwenExtractor created in {create_duration:.3f}s")
    return extractor

# Legacy compatibility for DocumentProcessor
VALID_QWEN_MODELS = {
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
}

def get_model_name_from_size(size: str) -> str:
    return f"Qwen/Qwen2.5-{size}-Instruct"

@dataclass
class TextGenConfig:
    max_new_tokens: int = 300
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.03
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

################################################################################
# Additional Helper Classes for DocumentProcessor Compatibility
################################################################################

class ComprehensiveInfoExtractor:
    """Backwards compatibility wrapper for DocumentProcessor"""
    
    def __init__(self):
        comp_init_start_time = get_ph_time()
        self.config = ExtractionConfig(fast_mode=True)
        self.fast_extractor = FastInfoExtractor(self.config)
        
        comp_init_complete_time = get_ph_time()
        comp_init_duration = (comp_init_complete_time - comp_init_start_time).total_seconds()
        logger.info(f"[{format_ph_time(comp_init_complete_time)}] ComprehensiveInfoExtractor initialized in {comp_init_duration:.3f}s (compatibility mode)")
    
    def detect_document_type_enhanced(self, text: str, filename: str) -> tuple:
        """Compatibility method for DocumentProcessor"""
        return self.fast_extractor.detect_document_type_fast(text, filename)
    
    def extract_comprehensive_metadata(self, text: str, doc_type: str, filename: str) -> dict:
        """Compatibility method for DocumentProcessor"""
        return self.fast_extractor.extract_fast_metadata(text, doc_type, filename)
    
    def calculate_enhanced_confidence(self, result: dict, text: str) -> float:
        """Compatibility method for DocumentProcessor"""
        return self.fast_extractor._calculate_fast_confidence(result, text, "")

################################################################################
# Performance Monitoring for DocumentProcessor Integration
################################################################################

class ExtractionPerformanceMonitor:
    """Monitor extraction performance for DocumentProcessor integration"""
    
    def __init__(self):
        monitor_init_start_time = get_ph_time()
        self.stats = {
            'extractions_count': 0,
            'total_time': 0.0,
            'fast_extractions': 0,
            'ai_extractions': 0,
            'fallback_extractions': 0,
            'average_confidence': 0.0
        }
        self.stats_lock = threading.Lock()
        
        monitor_init_complete_time = get_ph_time()
        monitor_init_duration = (monitor_init_complete_time - monitor_init_start_time).total_seconds()
        logger.debug(f"[{format_ph_time(monitor_init_complete_time)}] ExtractionPerformanceMonitor initialized in {monitor_init_duration:.3f}s")
    
    def record_extraction(self, processing_time: float, method: str, confidence: float):
        """Record extraction performance"""
        record_time = get_ph_time()
        
        with self.stats_lock:
            self.stats['extractions_count'] += 1
            self.stats['total_time'] += processing_time
            
            if method.startswith('fast'):
                self.stats['fast_extractions'] += 1
            elif method.startswith('ai'):
                self.stats['ai_extractions'] += 1
            else:
                self.stats['fallback_extractions'] += 1
            
            # Update average confidence
            current_avg = self.stats['average_confidence']
            count = self.stats['extractions_count']
            self.stats['average_confidence'] = ((current_avg * (count - 1)) + confidence) / count
        
        logger.debug(f"[{format_ph_time(record_time)}] Recorded extraction: {processing_time:.3f}s, {method}, {confidence:.2f}% confidence")
    
    def get_performance_report(self) -> dict:
        """Get performance report for DocumentProcessor"""
        with self.stats_lock:
            if self.stats['extractions_count'] == 0:
                return {'status': 'No extractions recorded'}
            
            avg_time = self.stats['total_time'] / self.stats['extractions_count']
            
            return {
                'total_extractions': self.stats['extractions_count'],
                'average_time_seconds': round(avg_time, 3),
                'extractions_per_minute': round(60 / avg_time, 1) if avg_time > 0 else 0,
                'method_breakdown': {
                    'fast_pattern': self.stats['fast_extractions'],
                    'ai_enhanced': self.stats['ai_extractions'],
                    'fallback': self.stats['fallback_extractions']
                },
                'average_confidence': round(self.stats['average_confidence'], 3),
                'total_processing_time': round(self.stats['total_time'], 2)
            }

# Global performance monitor instance
performance_monitor = ExtractionPerformanceMonitor()

################################################################################
# Main Integration Point for DocumentProcessor
################################################################################

def create_fast_extractor_for_document_processor(model_size: str = "7B") -> QwenExtractor:
    """
    Create a QwenExtractor specifically optimized for DocumentProcessor integration
    
    This function ensures:
    - Fast processing (pattern-first approach)
    - Memory efficient 
    - Compatible with DocumentProcessor's single-file workflow
    - Proper cleanup integration
    """
    create_start_time = get_ph_time()
    extractor = QwenExtractor(
        model_size=model_size,
        fast_mode=True,  # Enable fast mode for DocumentProcessor
        pattern_only_mode=False,  # Allow AI enhancement but prefer patterns
        config={
            'enable_caching': True,
            'ai_timeout_seconds': 15,  # Quick timeout for DocumentProcessor
            'include_debug_info': False
        }
    )
    create_complete_time = get_ph_time()
    create_duration = (create_complete_time - create_start_time).total_seconds()
    logger.info(f"[{format_ph_time(create_complete_time)}] Fast extractor for DocumentProcessor created in {create_duration:.3f}s")
    return extractor

# Export the main classes and functions needed by DocumentProcessor
__all__ = [
    'QwenExtractor',
    'ComprehensiveInfoExtractor', 
    'cleanup_qwen_models',
    'create_qwen_extractor',
    'create_fast_extractor_for_document_processor',
    'performance_monitor',
    'ExtractionConfig',
    'TextGenConfig'
]