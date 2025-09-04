# metadata.py - Enhanced with Qwen 2.5 Instruct for faster metadata extraction
# Uses Qwen 2.5 Instruct (7B) for text processing and Qwen 2VL for OCR
# All imports and function names maintained for compatibility

import json
import logging
import re
import os
import time
import threading
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import (
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################################################################
# Complete Information Extraction Engine
################################################################################

class SmartExtractor:
    """Smart information extraction with comprehensive patterns"""
    
    def __init__(self):
        self.doc_patterns = {
            'electronic_ticket': {
                'keywords': ['electronic ticket', 'ticket receipt', 'booking ref', 'passenger', 'flight', 'emd', 'miscellaneous document'],
                'confidence': 0.9
            },
            'official_receipt': {
                'keywords': ['official receipt', 'or number', 'received from', 'amount received', 'bir'],
                'confidence': 0.8
            },
            'invoice': {
                'keywords': ['invoice number', 'bill to', 'amount due', 'invoice date'],
                'confidence': 0.8
            },
            'voucher': {
                'keywords': ['voucher', 'payment voucher', 'check voucher'],
                'confidence': 0.7
            }
        }
        
    def detect_document_type(self, text: str, filename: str) -> str:
        """Detect document type from content and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        best_score = 0
        best_type = 'general'
        
        for doc_type, config in self.doc_patterns.items():
            score = 0
            
            # Check filename
            filename_matches = sum(1 for keyword in config['keywords'] if keyword.lower() in filename_lower)
            if filename_matches > 0:
                score += 0.4
            
            # Check content
            content_matches = sum(1 for keyword in config['keywords'] if keyword.lower() in text_lower)
            if content_matches >= 2:
                score += 0.6
            
            if score > best_score:
                best_score = score
                best_type = doc_type
        
        return best_type
    
    def extract_comprehensive_info(self, text: str, doc_type: str) -> dict:
        """Extract all possible information from text"""
        
        result = {
            "doc_type": doc_type,
            "title": self._extract_title(text, doc_type),
            "ids": self._extract_all_ids(text, doc_type),
            "parties": self._extract_parties(text),
            "dates": self._extract_all_dates(text),
            "amounts": self._extract_all_amounts(text),
            "line_items": [],
            "payment_terms": None,
            "addresses": {},
            "contacts": self._extract_contacts(text),
            "status": "processed",
            "tags": [doc_type],
            "extracted_entities": self._extract_entities(text),
            "notes": [],
            "confidence": self._calculate_confidence(text, doc_type)
        }
        
        return result
    
    def _extract_title(self, text: str, doc_type: str) -> str:
        """Extract meaningful document title"""
        lines = text.split('\n')
        
        # Look for title indicators
        title_keywords = ['receipt', 'ticket', 'invoice', 'electronic', 'document']
        
        for line in lines[:8]:
            line = line.strip()
            if (10 <= len(line) <= 80 and 
                any(keyword in line.lower() for keyword in title_keywords)):
                return line
        
        # Use document type with key information
        if doc_type == 'electronic_ticket':
            booking_ref = self._find_pattern(text, [r'booking\s+ref:?\s*([A-Z0-9]+)'])
            if booking_ref:
                return f"Electronic Ticket - {booking_ref}"
            return "Electronic Ticket Receipt"
        
        elif doc_type == 'official_receipt':
            or_num = self._find_pattern(text, [r'or\s*(?:no|number)?:?\s*([A-Z0-9-]+)'])
            if or_num:
                return f"Official Receipt {or_num}"
            return "Official Receipt"
        
        # Fallback to first meaningful line
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 8:
                return line
        
        return f"{doc_type.replace('_', ' ').title()}"
    
    def _extract_all_ids(self, text: str, doc_type: str) -> dict:
        """Extract all document identifiers"""
        ids = {}
        
        patterns = {
            'reference_no': [
                r'booking\s+ref:?\s*([A-Z0-9]{4,8})',
                r'reference:?\s*([A-Z0-9]{4,8})',
                r'ref:?\s*([A-Z0-9]{4,8})'
            ],
            'document_no': [
                r'ticket\s+number:?\s*([0-9\s]+)',
                r'document\s+number:?\s*([0-9\s]+)',
                r'or\s*(?:no|number)?:?\s*([A-Z0-9-]+)',
                r'invoice\s*(?:no|number)?:?\s*([A-Z0-9-]+)',
                r'voucher\s*(?:no|number)?:?\s*([A-Z0-9-]+)',
                r'(\d{3}\s+\d{10})'
            ]
        }
        
        for id_type, pattern_list in patterns.items():
            value = self._find_pattern(text, pattern_list)
            if value:
                ids[id_type] = value.strip()
        
        return ids
    
    def _extract_parties(self, text: str) -> dict:
        """Extract issuer and recipient information"""
        parties = {
            "issuer": {"name": None, "tin": None},
            "recipient": {"name": None, "tin": None},
            "other": []
        }
        
        # Extract companies (issuers)
        company_patterns = [
            r'(Philippine Airlines?)',
            r'(Cebu Pacific)',
            r'([A-Z][A-Z\s&]+(?:CORP|CORPORATION|INC|COMPANY|AIRLINES?))',
            r'issuing\s+office:\s*([A-Z][^,\n]+)',
        ]
        
        issuer_name = self._find_pattern(text, company_patterns)
        if issuer_name:
            parties['issuer']['name'] = issuer_name.strip()
        
        # Extract people (recipients)
        person_patterns = [
            r'passenger:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)\s*(?:Mr|Ms|Mrs|Dr)?',
            r'received\s+from:\s*([A-Z][^,\n]+)',
            r'bill\s+to:\s*([A-Z][^,\n]+)',
            r'([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*(?:Mr|Ms|Mrs|Dr|ADT)'
        ]
        
        recipient_name = self._find_pattern(text, person_patterns)
        if recipient_name:
            # Clean up the name
            name = ' '.join(word.capitalize() for word in recipient_name.strip().split())
            parties['recipient']['name'] = name
        
        return parties
    
    def _extract_all_dates(self, text: str) -> dict:
        """Extract all dates from document"""
        dates = {}
        
        date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            r'\b(\d{1,2}\s*[A-Za-z]{3,9}\s*\d{4})\b',
            r'\b([A-Za-z]{3,9}\s*\d{1,2},?\s*\d{4})\b',
            r'date:\s*(\d{1,2}[A-Za-z]{3}\d{4})',
            r'(\d{2}[A-Za-z]{3}\d{4})'
        ]
        
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            found_dates.extend(matches)
        
        # Remove duplicates and assign types
        unique_dates = list(dict.fromkeys(found_dates))
        
        if unique_dates:
            dates['issue_date'] = unique_dates[0]
            if len(unique_dates) > 1:
                dates['due_date'] = unique_dates[1]
        
        return dates
    
    def _extract_all_amounts(self, text: str) -> dict:
        """Extract all monetary amounts"""
        amounts = {"currency": None, "total": None, "subtotal": None, "tax": None}
        
        # Detect currency
        if any(symbol in text.lower() for symbol in ['₱', 'php', 'peso']):
            amounts["currency"] = "PHP"
        elif 'usd' in text.lower() or '$' in text:
            amounts["currency"] = "USD"
        else:
            amounts["currency"] = "PHP"  # Default
        
        # Extract amounts
        amount_patterns = [
            r'PHP\s*([\d,]+\.?\d*)',
            r'₱\s*([\d,]+\.?\d*)',
            r'USD\s*([\d,]+\.?\d*)',
            r'\$\s*([\d,]+\.?\d*)',
            r'total[:\s]+([\d,]+\.?\d*)',
            r'amount[:\s]+([\d,]+\.?\d*)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:PHP|₱)',
            r'/PHP\s*([\d,]+)',
            r'fare\s+equivalent:\s*PHP\s*([\d,]+)',
        ]
        
        found_amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    amount_str = match.replace(',', '')
                    amount = float(amount_str)
                    if amount > 0:
                        found_amounts.append(amount)
                except ValueError:
                    continue
        
        # Assign amounts (largest is usually total)
        if found_amounts:
            amounts['total'] = max(found_amounts)
            
            # Look for tax specifically
            tax_patterns = [r'tax[:\s]+([\d,]+\.?\d*)', r'vat[:\s]+([\d,]+\.?\d*)']
            for pattern in tax_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        amounts['tax'] = float(match.group(1).replace(',', ''))
                        break
                    except ValueError:
                        continue
        
        return amounts
    
    def _extract_contacts(self, text: str) -> dict:
        """Extract contact information"""
        contacts = {}
        
        # Phone numbers
        phone_patterns = [
            r'telephone:\s*([+\d\s\-()]+)',
            r'phone:\s*([+\d\s\-()]+)',
            r'(\+63[^,\n\s]{8,})',
            r'(\(\d{3}\)\s*\d{3}-\d{4})',
            r'(\d{2,3}-\d{4}-\d{4})',
        ]
        
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                phone = match.strip()
                if len(phone) > 6 and phone not in phones:
                    phones.append(phone)
        
        if phones:
            contacts['phone'] = phones[0]  # Take first one
        
        # Email addresses
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        emails = re.findall(email_pattern, text)
        if emails:
            contacts['email'] = emails[0]
        
        return contacts
    
    def _extract_entities(self, text: str) -> dict:
        """Extract entities for search functionality"""
        entities = {
            "people": [],
            "companies": [],
            "locations": [],
            "projects": []
        }
        
        # Extract people
        person_patterns = [
            r'passenger:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)\s*(?:Mr|Ms|Mrs|Dr)?',
            r'([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*(?:Mr|Ms|Mrs|Dr|ADT)'
        ]
        
        for pattern in person_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = ' '.join(word.capitalize() for word in match.strip().split())
                if len(name.split()) >= 2 and name not in entities["people"]:
                    entities["people"].append(name)
        
        # Extract companies
        company_patterns = [
            r'(Philippine Airlines?)',
            r'(Cebu Pacific)',
            r'([A-Z][A-Z\s&]+(?:CORP|CORPORATION|INC|COMPANY|AIRLINES?))'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                company = match.strip()
                if company and company not in entities["companies"]:
                    entities["companies"].append(company)
        
        # Extract locations
        location_patterns = [
            r'([A-Z\s]+INTERNATIONAL)\s*(?:AIRPORT)?',
            r'([A-Z\s]+INTL)\s*(?:AIRPORT)?',
            r'from:\s*([^|]+?)\s+to:\s*([^|]+?)(?:\s+flight|$)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for location in match:
                        location = location.strip()
                        if location and location not in entities["locations"]:
                            entities["locations"].append(location)
                else:
                    location = match.strip()
                    if location and location not in entities["locations"]:
                        entities["locations"].append(location)
        
        return entities
    
    def _find_pattern(self, text: str, patterns: list) -> Optional[str]:
        """Find first match from list of patterns"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _calculate_confidence(self, text: str, doc_type: str) -> float:
        """Calculate extraction confidence"""
        confidence = 0.5  # Base confidence
        
        # Text length bonus
        if len(text) > 300:
            confidence += 0.1
        
        # Structure indicators
        indicators = ['total', 'amount', 'date', 'number', 'name', 'flight', 'passenger']
        found_indicators = sum(1 for ind in indicators if ind.lower() in text.lower())
        confidence += (found_indicators / len(indicators)) * 0.3
        
        # Document type specific bonuses
        if doc_type != 'general':
            confidence += 0.1
        
        return min(confidence, 1.0)

################################################################################
# Qwen 2.5 Instruct Model Manager - Optimized for Text Processing
################################################################################

class Qwen25ModelManager:
    """Shared model manager for Qwen 2.5 Instruct text extraction models"""
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        
        if self.device == "cuda":
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Qwen25ModelManager using GPU with {memory_gb:.1f}GB memory")
        else:
            logger.info("Qwen25ModelManager using CPU")
    
    def get_qwen25_text_model(self, model_size="7B"):
        """Get Qwen 2.5 Instruct model for text extraction - much faster than Qwen 2VL"""
        model_key = f"qwen25_instruct_{model_size}"
        
        if model_key not in self.models:
            with self._lock:
                if model_key not in self.models:
                    try:
                        logger.info(f"Loading Qwen 2.5 Instruct {model_size} for fast text extraction")
                        model_name = f"Qwen/Qwen2.5-{model_size}-Instruct"
                        
                        # Optimized quantization for speed
                        quant = None
                        if self.device == "cuda":
                            quant = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                            )
                        
                        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="auto" if self.device == "cuda" else None,
                            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                            quantization_config=quant,
                            trust_remote_code=True,
                            # Speed optimizations
                            use_cache=True,
                            low_cpu_mem_usage=True,
                        )
                        
                        if self.device == "cpu":
                            model = model.to("cpu")
                        
                        model.eval()
                        
                        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is None:
                            tokenizer.pad_token_id = tokenizer.eos_token_id
                        
                        self.models[model_key] = (model, tokenizer)
                        logger.info(f"Qwen 2.5 Instruct {model_size} text extraction model loaded successfully")
                        
                    except Exception as e:
                        logger.error(f"Failed to load Qwen 2.5 Instruct model: {e}")
                        self.models[model_key] = (None, None)
        
        return self.models.get(model_key, (None, None))
    
    def cleanup(self):
        """Cleanup resources"""
        with self._lock:
            for key, value in self.models.items():
                if value and value != (None, None):
                    model, tokenizer = value
                    if model is not None:
                        del model
                    if tokenizer is not None:
                        del tokenizer
            self.models.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Qwen25ModelManager cleanup completed")

################################################################################
# Enhanced QwenExtractor - Now using Qwen 2.5 Instruct for Speed
################################################################################

class QwenExtractor:
    """
    Enhanced QwenExtractor using Qwen 2.5 Instruct for faster metadata extraction
    100% compatible with existing document_processor.py
    All method names and interfaces maintained exactly
    """

    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        config: Optional[dict] = None,
        model_size: str = "7B",
        **kwargs
    ):
        self.config = config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        
        # Initialize smart extractor for comprehensive pattern matching
        self.smart_extractor = SmartExtractor()
        
        # Initialize Qwen 2.5 Instruct model manager (faster than Qwen 2VL)
        self.model_manager = Qwen25ModelManager()
        
        # Get Qwen 2.5 Instruct model and tokenizer from shared manager
        self.model, self.tokenizer = self.model_manager.get_qwen25_text_model(self.model_size)
        
        if self.model is None or self.tokenizer is None:
            logger.warning(f"QwenExtractor using pattern extraction only (AI unavailable)")
            self.use_ai = False
        else:
            self.use_ai = True
            # Optimized generation config for speed
            self.generation_config = GenerationConfig(
                max_new_tokens=600,  # Reduced for speed
                do_sample=False,     # Greedy decoding for speed
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.02,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,      # Speed optimization
            )
        
        logger.info(f"QwenExtractor initialized - Model: Qwen 2.5 Instruct {self.model_size}, AI: {self.use_ai}")

    def extract_json(self, ocr_text: str, doc_type: str = None) -> Dict[str, Any]:
        """
        Extract structured data from OCR text using Qwen 2.5 Instruct for speed
        EXACT SAME INTERFACE as original - maintains full compatibility
        """
        try:
            # Detect document type if not provided
            if not doc_type or doc_type == 'general':
                detected_type = self.smart_extractor.detect_document_type(ocr_text or "", doc_type or "")
            else:
                detected_type = doc_type
            
            # Use comprehensive extraction (always works with patterns)
            comprehensive_data = self.smart_extractor.extract_comprehensive_info(ocr_text or "", detected_type)
            
            # Enhance with Qwen 2.5 Instruct if available (much faster than Qwen 2VL)
            if self.use_ai:
                try:
                    ai_data = self._ai_enhance_extraction_fast(ocr_text, detected_type)
                    if ai_data:
                        comprehensive_data = self._merge_ai_data(comprehensive_data, ai_data)
                        comprehensive_data['processing_info'] = {
                            'model_used': f'Qwen-2.5-Instruct-{self.model_size}',
                            'extraction_method': 'ai_enhanced_fast'
                        }
                except Exception as e:
                    logger.warning(f"AI enhancement failed, using pattern extraction: {e}")
                    comprehensive_data['processing_info'] = {
                        'model_used': 'pattern_only',
                        'extraction_method': 'pattern_fallback'
                    }
            else:
                comprehensive_data['processing_info'] = {
                    'model_used': 'pattern_only', 
                    'extraction_method': 'pattern_only'
                }
            
            # Always return legacy compatible format
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error in extract_json: {e}")
            return self._create_fallback_result(doc_type or "general")

    @torch.inference_mode()
    def _ai_enhance_extraction_fast(self, ocr_text: str, doc_type: str) -> dict:
        """Use Qwen 2.5 Instruct to enhance extraction - optimized for speed"""
        try:
            # Truncate text more aggressively for speed
            max_text_length = 2000 if self.device == "cuda" else 1500
            truncated_text = ocr_text[:max_text_length] if len(ocr_text) > max_text_length else ocr_text
            
            prompt = self._build_fast_ai_prompt(truncated_text, doc_type)
            
            # Tokenize with padding disabled for speed
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=1024,  # Shorter context for speed
                padding=False
            ).to(self.device)
            
            # Fast generation
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                output_ids = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    use_cache=True
                )
            
            # Decode only generated tokens
            generated_tokens = output_ids[0][len(inputs.input_ids[0]):]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return self._parse_ai_response(text)
            
        except Exception as e:
            logger.warning(f"Fast AI enhancement error: {e}")
            return {}

    def _build_fast_ai_prompt(self, ocr_text: str, doc_type: str) -> str:
        """Build optimized AI prompt for fast extraction"""
        
        # Document type specific prompts for better accuracy
        if doc_type == 'electronic_ticket':
            prompt_template = """Extract key info from this airline ticket. Return JSON:
{
  "issuer": "airline name",
  "passenger": "passenger name", 
  "booking_ref": "booking reference",
  "ticket_no": "ticket number",
  "amount": "total amount",
  "currency": "PHP/USD",
  "date": "issue date",
  "route": "from-to"
}

Document: {text}

JSON:"""
        elif doc_type == 'official_receipt':
            prompt_template = """Extract key info from this official receipt. Return JSON:
{
  "issuer": "company name",
  "recipient": "received from",
  "or_number": "OR number",
  "amount": "total amount",
  "currency": "PHP/USD", 
  "date": "receipt date",
  "description": "purpose"
}

Document: {text}

JSON:"""
        else:
            prompt_template = """Extract key information from this document. Return JSON:
{
  "title": "document title",
  "issuer": "issuing party",
  "recipient": "receiving party",
  "amount": "monetary amount",
  "currency": "currency",
  "date": "document date",
  "reference": "reference number"
}

Document: {text}

JSON:"""
        
        return prompt_template.format(text=ocr_text)

    def _parse_ai_response(self, text: str) -> dict:
        """Parse AI JSON response safely"""
        try:
            # Find JSON in response
            start = text.find('{')
            if start == -1:
                return {}
            
            end = text.rfind('}') + 1
            if end <= start:
                return {}
            
            json_str = text[start:end]
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            try:
                # Remove trailing commas
                fixed_json = re.sub(r',\s*}', '}', json_str)
                fixed_json = re.sub(r',\s*]', ']', fixed_json)
                return json.loads(fixed_json)
            except:
                logger.warning(f"AI JSON parsing failed: {e}")
                return {}
        except Exception as e:
            logger.warning(f"AI response parsing error: {e}")
            return {}

    def _merge_ai_data(self, comprehensive_data: dict, ai_data: dict) -> dict:
        """Merge AI enhancement with comprehensive extraction"""
        result = comprehensive_data.copy()
        
        # Map AI fields to comprehensive structure
        field_mappings = {
            'issuer': ('parties', 'issuer', 'name'),
            'passenger': ('parties', 'recipient', 'name'),
            'recipient': ('parties', 'recipient', 'name'),
            'booking_ref': ('ids', 'reference_no'),
            'or_number': ('ids', 'reference_no'),
            'ticket_no': ('ids', 'document_no'),
            'reference': ('ids', 'reference_no'),
            'amount': ('amounts', 'total'),
            'currency': ('amounts', 'currency'),
            'date': ('dates', 'issue_date'),
            'title': ('title',),
        }
        
        for ai_field, target_path in field_mappings.items():
            if ai_data.get(ai_field):
                self._set_nested_value(result, target_path, ai_data[ai_field])
        
        # Improve confidence if AI provided good data
        if any(ai_data.get(key) for key in ['issuer', 'amount', 'date', 'reference']):
            result['confidence'] = min(result['confidence'] + 0.2, 1.0)
            
        return result
    
    def _set_nested_value(self, data: dict, path: tuple, value: Any):
        """Set nested dictionary value"""
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _create_fallback_result(self, doc_type: str) -> dict:
        """Create minimal fallback result - maintains exact format compatibility"""
        return {
            "doc_type": doc_type,
            "title": None,
            "ids": {},
            "parties": {"issuer": {"name": None, "tin": None}, "recipient": {"name": None, "tin": None}, "other": []},
            "dates": {},
            "amounts": {},
            "line_items": [],
            "payment_terms": None,
            "addresses": {},
            "contacts": {},
            "status": "processed",
            "tags": [doc_type],
            "extracted_entities": {"people": [], "companies": [], "locations": [], "projects": []},
            "notes": [],
            "confidence": 0.3
        }

    def cleanup(self):
        """Clean up resources - exact same interface"""
        try:
            self.model = None
            self.tokenizer = None
            logger.info("QwenExtractor cleanup completed")
        except Exception as e:
            logger.warning(f"QwenExtractor cleanup warning: {e}")

################################################################################
# Backward Compatibility - Legacy Qwen 2VL Model Manager
################################################################################

class QwenModelManager:
    """Legacy compatibility class - redirects to Qwen 2.5 Instruct for speed"""
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
        # Redirect to Qwen 2.5 Instruct manager for speed
        self.qwen25_manager = Qwen25ModelManager()
        logger.info("QwenModelManager (legacy) redirecting to Qwen 2.5 Instruct for speed")
    
    def get_qwen_text_model(self, model_size="7B"):
        """Legacy method - redirects to faster Qwen 2.5 Instruct"""
        return self.qwen25_manager.get_qwen25_text_model(model_size)
    
    def cleanup(self):
        """Legacy cleanup method"""
        self.qwen25_manager.cleanup()

################################################################################
# Required Functions for Compatibility
################################################################################

def cleanup_qwen_models():
    """
    Global cleanup function - FIXES THE IMPORT ERROR
    This function was missing and causing the ImportError
    """
    try:
        # Cleanup both managers
        QwenModelManager().cleanup()
        Qwen25ModelManager().cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Global Qwen models cleanup completed")
    except Exception as e:
        logger.warning(f"Qwen models cleanup failed: {e}")

def create_qwen_extractor(model_size: str = "7B", **kwargs) -> QwenExtractor:
    """Factory function to create QwenExtractor with Qwen 2.5 Instruct"""
    return QwenExtractor(model_size=model_size, **kwargs)

# Legacy compatibility exports
VALID_QWEN_MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct", 
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
    "32B": "Qwen/Qwen2.5-32B-Instruct",
    "72B": "Qwen/Qwen2.5-72B-Instruct",
}

def get_model_name_from_size(size: str) -> str:
    """Legacy function - now returns Qwen 2.5 Instruct for speed"""
    return f"Qwen/Qwen2.5-{size}-Instruct"

# Additional compatibility classes that might be needed
@dataclass
class TextGenConfig:
    max_new_tokens: int = 600  # Reduced for speed
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.02
    stop_strings: Tuple[str, ...] = ("</s>",)
    load_4bit: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

################################################################################
# Performance Monitoring
################################################################################

class ExtractionMetrics:
    """Track extraction performance for optimization"""
    
    def __init__(self):
        self.metrics = {
            'total_extractions': 0,
            'ai_enhanced': 0,
            'pattern_only': 0,
            'avg_time_ai': 0.0,
            'avg_time_pattern': 0.0,
            'confidence_scores': []
        }
        self.lock = threading.Lock()
    
    def record_extraction(self, method: str, time_taken: float, confidence: float):
        """Record extraction metrics"""
        with self.lock:
            self.metrics['total_extractions'] += 1
            self.metrics['confidence_scores'].append(confidence)
            
            if method.startswith('ai'):
                self.metrics['ai_enhanced'] += 1
                # Running average
                prev_avg = self.metrics['avg_time_ai']
                count = self.metrics['ai_enhanced']
                self.metrics['avg_time_ai'] = (prev_avg * (count - 1) + time_taken) / count
            else:
                self.metrics['pattern_only'] += 1
                prev_avg = self.metrics['avg_time_pattern'] 
                count = self.metrics['pattern_only']
                self.metrics['avg_time_pattern'] = (prev_avg * (count - 1) + time_taken) / count
    
    def get_summary(self) -> dict:
        """Get performance summary"""
        with self.lock:
            if self.metrics['confidence_scores']:
                avg_confidence = sum(self.metrics['confidence_scores']) / len(self.metrics['confidence_scores'])
            else:
                avg_confidence = 0.0
                
            return {
                'total_extractions': self.metrics['total_extractions'],
                'ai_enhanced_count': self.metrics['ai_enhanced'],
                'pattern_only_count': self.metrics['pattern_only'],
                'avg_time_ai_seconds': round(self.metrics['avg_time_ai'], 2),
                'avg_time_pattern_seconds': round(self.metrics['avg_time_pattern'], 2),
                'avg_confidence': round(avg_confidence, 2),
                'model_used': 'Qwen-2.5-Instruct-7B'
            }

# Global metrics instance
extraction_metrics = ExtractionMetrics()

################################################################################
# Speed Testing Utilities
################################################################################

def test_extraction_speed(sample_text: str, iterations: int = 10) -> dict:
    """Test extraction speed with Qwen 2.5 Instruct vs patterns"""
    extractor = QwenExtractor()
    
    # Test pattern-only extraction
    pattern_times = []
    extractor.use_ai = False
    
    for i in range(iterations):
        start = time.time()
        extractor.extract_json(sample_text, "invoice")
        pattern_times.append(time.time() - start)
    
    # Test AI-enhanced extraction
    ai_times = []
    extractor.use_ai = True
    
    if extractor.model is not None:
        for i in range(iterations):
            start = time.time()
            extractor.extract_json(sample_text, "invoice")
            ai_times.append(time.time() - start)
    
    return {
        'pattern_only_avg': round(sum(pattern_times) / len(pattern_times), 3) if pattern_times else 0,
        'ai_enhanced_avg': round(sum(ai_times) / len(ai_times), 3) if ai_times else 0,
        'speedup_factor': round((sum(ai_times) / len(ai_times)) / (sum(pattern_times) / len(pattern_times)), 2) if ai_times and pattern_times else 0,
        'model_used': 'Qwen-2.5-Instruct-7B'
    }