import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class QwenExtractor:
    def __init__(self, device=None, force_cpu=False):
        """Streamlined Qwen extractor for document processing"""
        self.device = self._setup_device(device, force_cpu)
        self.model = None
        self.tokenizer = None
        self.fallback_mode = False
        
        # Initialize model
        self._initialize_model()

    def _setup_device(self, device, force_cpu):
        """Setup device"""
        if force_cpu:
            return "cpu"
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self):
        """Initialize Qwen model"""
        try:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            logger.info(f"Loading Qwen model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if self.device == "cpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )

            self.model.eval()
            logger.info("Qwen model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qwen model: {e}")
            self.fallback_mode = True

    def extract_structured_data(self, text: str, category: str = None) -> Dict[str, Any]:
        """Main extraction method"""
        try:
            if self.fallback_mode:
                return self._pattern_extraction(text)
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Try ML extraction first
            ml_result = self._ml_extraction(cleaned_text)
            
            # Always do pattern extraction as backup
            pattern_result = self._pattern_extraction(cleaned_text)
            
            # Merge results, preferring pattern extraction for reliability
            merged_data = self._merge_results(pattern_result, ml_result)
            
            # Calculate confidence
            confidence = self._calculate_confidence(merged_data, cleaned_text)
            
            return {
                'data': merged_data,
                'confidence': confidence,
                'method': 'hybrid_extraction',
                'document_type': self._detect_document_type(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return self._pattern_extraction(text)

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve structure
        cleaned = re.sub(r'\n\s*\n', '\n', text)  # Remove empty lines
        cleaned = re.sub(r' +', ' ', cleaned)      # Multiple spaces to single
        return cleaned.strip()

    def _ml_extraction(self, text: str) -> Dict[str, Any]:
        """Extract using Qwen model"""
        try:
            prompt = f"""Extract key information from this document. Return only valid JSON.

Document:
{text[:2000]}

Extract these fields if present:
- document_type: type of document
- company_name: company or organization name
- payee: who receives payment
- date: any dates found
- amount: monetary amounts
- purpose: description or purpose
- batch_number: batch/reference numbers
- prepared_by: who prepared document
- approved_by: who approved document

Return JSON only:"""

            # Generate response
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1200,
                padding=False
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Parse JSON
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            return {}
            
        except Exception as e:
            logger.error(f"ML extraction failed: {e}")
            return {}

    def _pattern_extraction(self, text: str) -> Dict[str, Any]:
        """High-accuracy pattern-based extraction"""
        data = {}
        text_upper = text.upper()
        
        # Document type detection
        if 'REQUEST FOR PAYMENT' in text_upper:
            data['document_type'] = 'payment_request'
        elif 'VOUCHER' in text_upper:
            data['document_type'] = 'voucher'
        elif 'INVOICE' in text_upper:
            data['document_type'] = 'invoice'
        elif 'RECEIPT' in text_upper:
            data['document_type'] = 'receipt'
        else:
            data['document_type'] = 'general'

        # Company name extraction - look for known companies first
        company_patterns = [
            r'(COMFAC(?:\s+(?:CORPORATION|CORP|GLOBAL))?)',
            r'(EEI(?:\s+(?:CORPORATION|CORP))?)',
            r'(CORNERSTE+L(?:\s+(?:CORPORATION|CORP))?)',
            r'(ESCO(?:\s+(?:CORPORATION|CORP))?)',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text_upper)
            if match:
                data['company_name'] = match.group(1).strip()
                break

        # Extract structured fields based on document layout
        field_patterns = {
            'payee': [
                r'PAYEE[:\s]*([A-Za-z\s\.&,]+?)(?:\s*DATE|$)',
                r'PAID\s+TO[:\s]*([A-Za-z\s\.&,]+?)(?:\n|$)',
            ],
            'date': [
                r'DATE[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            ],
            'amount': [
                r'AMOUNT[:\s]*(?:PHP|USD|\$|₱)?\s*([\d,]+\.?\d*)',
                r'TOTAL[:\s]*(?:PHP|USD|\$|₱)?\s*([\d,]+\.?\d*)',
                r'(?:PHP|USD|\$|₱)\s*([\d,]+\.?\d*)',
            ],
            'purpose': [
                r'PURPOSE[\/\s]*DESCRIPTION[:\s]*([A-Za-z\s\.,\-]+?)(?:\n[A-Z]|\s*AMOUNT|$)',
                r'DESCRIPTION[:\s]*([A-Za-z\s\.,\-]+?)(?:\n[A-Z]|\s*AMOUNT|$)',
            ],
            'batch_number': [
                r'BATCH\s*(?:NO|NUMBER)[:\s]*([A-Z0-9\-]+)',
                r'(?:REF|REFERENCE)[:\s]*([A-Z0-9\-]+)',
            ],
            'prepared_by': [
                r'PREPARED\s*BY[:\s]*([A-Za-z\s\.]+?)(?:\n|RECOMMEND)',
            ],
            'approved_by': [
                r'APPROVED\s*BY[:\s]*([A-Za-z\s\.]+?)(?:\n|PAYMENT)',
            ],
            'amount_in_words': [
                r'AMOUNT\s*IN\s*WORDS[:\s]*([A-Za-z\s]+?)(?:\n[A-Z]|$)',
            ]
        }

        for field_name, patterns in field_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    if self._is_valid_value(value):
                        data[field_name] = value
                        break

        # Extract numbers that might be amounts
        if 'amount' not in data:
            number_matches = re.findall(r'\b(\d{3,}(?:,\d{3})*(?:\.\d{2})?)\b', text)
            if number_matches:
                # Take the largest number as likely amount
                amounts = [float(num.replace(',', '')) for num in number_matches]
                if amounts:
                    data['amount'] = f"{max(amounts):,.2f}"

        return self._clean_extracted_data(data)

    def _is_valid_value(self, value: str) -> bool:
        """Check if extracted value is meaningful"""
        if not value or len(value.strip()) < 2:
            return False
        
        # Skip common OCR artifacts and meaningless text
        invalid_patterns = [
            r'^[^\w\s]+$',  # Only special characters
            r'^\s*$',       # Only whitespace
            r'^[\.\-_]+$',  # Only dots, dashes, underscores
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, value):
                return False
                
        return True

    def _clean_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up extracted data"""
        cleaned = {}
        
        for key, value in data.items():
            if not value:
                continue
                
            if isinstance(value, str):
                # Clean up the value
                value = re.sub(r'\s+', ' ', value.strip())
                value = re.sub(r'[^\w\s\.\-@,]', '', value)
                
                # Skip if too short or meaningless
                if len(value) < 2 or value.lower() in ['na', 'n/a', 'null', 'none']:
                    continue
                
                cleaned[key] = value
            else:
                cleaned[key] = value
        
        return cleaned

    def _merge_results(self, pattern_result: Dict, ml_result: Dict) -> Dict[str, Any]:
        """Merge pattern and ML results, preferring pattern for reliability"""
        merged = pattern_result.copy()
        
        # Add ML results only if pattern didn't find them and they look valid
        for key, value in ml_result.items():
            if key not in merged and value and self._is_valid_value(str(value)):
                merged[key] = value
        
        return merged

    def _detect_document_type(self, text: str) -> str:
        """Detect document type"""
        text_upper = text.upper()
        
        if 'REQUEST FOR PAYMENT' in text_upper:
            return 'payment_request'
        elif 'VOUCHER' in text_upper:
            return 'voucher'
        elif 'INVOICE' in text_upper:
            return 'invoice'
        elif 'RECEIPT' in text_upper:
            return 'receipt'
        else:
            return 'general'

    def _calculate_confidence(self, data: Dict[str, Any], text: str) -> float:
        """Calculate extraction confidence"""
        if not data:
            return 0.0
        
        base_confidence = 0.6
        
        # Field count bonus
        meaningful_fields = len([v for v in data.values() if v and str(v).strip()])
        field_bonus = min(meaningful_fields * 0.05, 0.3)
        
        # Document type specific bonus
        doc_type = data.get('document_type', 'general')
        if doc_type != 'general':
            field_bonus += 0.1
        
        return min(base_confidence + field_bonus, 0.95)

    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'model') and self.model:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer:
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass