import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import time

logger = logging.getLogger(__name__)

class QwenExtractor:
    def __init__(self, device=None, force_cpu=False, model_size="1.5B"):
        """Advanced Qwen extractor with semantic understanding"""
        self.device = self._setup_device(device, force_cpu)
        self.model = None
        self.tokenizer = None
        self.fallback_mode = False
        self.model_size = model_size
        
        # Business intelligence patterns
        self.business_patterns = self._initialize_business_patterns()
        self._initialize_model()

    def _setup_device(self, device, force_cpu):
        if force_cpu:
            return "cpu"
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self):
        """Initialize Qwen model with proper size selection"""
        try:
            # Select model based on size parameter and available memory
            model_options = {
                "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
                "1.5B": "Qwen/Qwen2.5-1.5B-Instruct", 
                "3B": "Qwen/Qwen2.5-3B-Instruct",
                "7B": "Qwen/Qwen2.5-7B-Instruct",
                "14B": "Qwen/Qwen2.5-14B-Instruct"
            }
            
            model_name = model_options.get(self.model_size, model_options["1.5B"])
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
                # Use 4-bit quantization for larger models
                if self.model_size in ["1.5B", "14B"]:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
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
            logger.info(f"Qwen {self.model_size} model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qwen model: {e}")
            self.fallback_mode = True

    def _initialize_business_patterns(self) -> Dict[str, Any]:
        """Initialize business-specific patterns for document understanding"""
        return {
            'document_types': {
                'payment_request': [
                    r'request\s*for\s*payment',
                    r'payment\s*request',
                    r'prepared\s*by.*approved\s*by',
                ],
                'voucher': [
                    r'voucher\s*payable',
                    r'check\s*voucher',
                    r'v\.?\s*p\.?\s*no',
                    r'batch.*number'
                ],
                'email': [
                    r'from\s*:.*@',
                    r'to\s*:.*@',
                    r'subject\s*:',
                    r'sent\s*:'
                ],
                'check': [
                    r'pay\s*to\s*the\s*order\s*of',
                    r'check\s*number',
                    r'routing\s*number',
                    r'account\s*number'
                ],
                'transaction_slip': [
                    r'transaction\s*slip',
                    r'universal\s*transaction',
                    r'deposit\s*slip',
                    r'withdrawal\s*slip'
                ]
            },
            'companies': [
                r'COMFAC(?:\s+(?:CORPORATION|CORP|GLOBAL|GROUP))*',
                r'EEI(?:\s+(?:CORPORATION|CORP))*', 
                r'CORNERSTE{1,2}L(?:\s+(?:CORPORATION|CORP))*',
                r'ESCO(?:\s+(?:CORPORATION|CORP))*'
            ],
            'amounts': [
                r'(?:PHP|₱|USD|\$)\s*([\d,]+\.?\d*)',
                r'\b([\d,]+\.\d{2})\b(?=\s*(?:PHP|USD|$|\n))',
                r'\b(\d{1,3}(?:,\d{3})+)\b(?=\s*(?:PHP|USD|$|\n))'
            ],
            'dates': [
                r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\b',
                r'\b(\d{4}-\d{2}-\d{2})\b',
                r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})\b'
            ],
            'emails': [
                r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
            ],
            'phones': [
                r'\b(\+?63\s?[0-9\s\-\(\)]{10,})\b',
                r'\b(\+?1\s?[0-9\s\-\(\)]{10,})\b',
                r'\b(\(\d{3}\)\s?\d{3}[-.]?\d{4})\b'
            ],
            'reference_numbers': [
                r'\b([A-Z]{2,}\d{6,})\b',
                r'\b(\d{6,}[A-Z]{2,})\b',
                r'(?:ref|reference|batch|voucher|check)[\s#:]*([A-Z0-9\-]{6,})',
                r'\b(VP\d{6,})\b',
                r'\b(CK\d{6,})\b'
            ]
        }

    def extract_structured_data(self, text: str, category: str = None, file_info: Dict = None) -> Dict[str, Any]:
        """Main extraction with business intelligence"""
        start_time = time.time()
        
        try:
            # Clean and analyze text
            cleaned_text = self._smart_clean_text(text)
            doc_analysis = self._analyze_document_semantics(cleaned_text)
            
            # Extract with business context
            if self.fallback_mode:
                extracted_data = self._business_pattern_extraction(cleaned_text, doc_analysis)
            else:
                # Use ML with business context
                ml_data = self._semantic_ml_extraction(cleaned_text, doc_analysis)
                pattern_data = self._business_pattern_extraction(cleaned_text, doc_analysis)
                extracted_data = self._intelligent_merge(ml_data, pattern_data, doc_analysis)
            
            # Create structured result
            result = self._create_business_result(extracted_data, doc_analysis, cleaned_text)
            
            # Add processing metadata
            processing_time = time.time() - start_time
            result['processing_time_seconds'] = processing_time
            
            if file_info:
                result['processing_info'] = {
                    'category': category or 'unknown',
                    'classification': 'successful',
                    'document_name': file_info.get('document_name', ''),
                    'original_filename': file_info.get('original_filename', ''),
                    'processed_at': datetime.now().isoformat(),
                    'extraction_method': 'qwen_business_intelligence',
                    'model_size': self.model_size
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Business extraction failed: {e}")
            return self._create_fallback_result(text, file_info)

    def _smart_clean_text(self, text: str) -> str:
        """Smart text cleaning that preserves business context"""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve document structure
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 empty lines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)  # Remove leading spaces on lines
        
        # Fix common OCR issues that break business data
        cleaned = re.sub(r'(\d),(\d)', r'\1,\2', cleaned)  # Fix broken comma numbers
        cleaned = re.sub(r'(\d)\.(\d{2})\b', r'\1.\2', cleaned)  # Fix decimal amounts
        cleaned = re.sub(r'@([a-zA-Z])', r'@\1', cleaned)  # Fix broken emails
        
        return cleaned.strip()

    def _analyze_document_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze document for business semantics"""
        analysis = {
            'document_type': 'general',
            'confidence': 0.0,
            'business_entities': {},
            'key_sections': {},
            'data_quality': 'low'
        }
        
        text_upper = text.upper()
        
        # Detect document type with confidence
        type_scores = {}
        for doc_type, patterns in self.business_patterns['document_types'].items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_upper, re.IGNORECASE))
                score += matches * 2 if matches else 0
            
            if score > 0:
                type_scores[doc_type] = score
        
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            analysis['document_type'] = best_type
            analysis['confidence'] = min(type_scores[best_type] / 10.0, 1.0)
        
        # Identify key business sections
        analysis['key_sections'] = self._identify_business_sections(text, analysis['document_type'])
        
        # Assess data quality
        if len([line for line in text.split('\n') if line.strip()]) > 5:
            analysis['data_quality'] = 'high'
        elif len(text) > 200:
            analysis['data_quality'] = 'medium'
        
        return analysis

    def _identify_business_sections(self, text: str, doc_type: str) -> Dict[str, str]:
        """Identify key business sections in document"""
        sections = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if doc_type == 'payment_request':
            sections = self._map_payment_sections(lines)
        elif doc_type == 'email':
            sections = self._map_email_sections(lines)
        elif doc_type == 'voucher':
            sections = self._map_voucher_sections(lines)
        elif doc_type == 'check':
            sections = self._map_check_sections(lines)
        else:
            sections['full_content'] = '\n'.join(lines)
        
        return sections

    def _map_payment_sections(self, lines: List[str]) -> Dict[str, str]:
        """Map payment request sections"""
        sections = {}
        current_section = 'header'
        section_lines = []
        
        for line in lines:
            line_upper = line.upper()
            
            if 'COMPANY' in line_upper or 'PAYEE' in line_upper:
                sections[current_section] = '\n'.join(section_lines)
                current_section = 'payee_section'
                section_lines = [line]
            elif 'AMOUNT' in line_upper and 'WORDS' not in line_upper:
                sections[current_section] = '\n'.join(section_lines)
                current_section = 'amount_section'
                section_lines = [line]
            elif 'PURPOSE' in line_upper or 'DESCRIPTION' in line_upper:
                sections[current_section] = '\n'.join(section_lines)
                current_section = 'purpose_section'
                section_lines = [line]
            elif any(phrase in line_upper for phrase in ['PREPARED BY', 'APPROVED BY', 'RECEIVED BY']):
                sections[current_section] = '\n'.join(section_lines)
                current_section = 'approval_section'
                section_lines = [line]
            else:
                section_lines.append(line)
        
        sections[current_section] = '\n'.join(section_lines)
        return sections

    def _map_email_sections(self, lines: List[str]) -> Dict[str, str]:
        """Map email sections"""
        sections = {}
        header_section = []
        body_section = []
        in_header = True
        
        for line in lines:
            if re.match(r'^(From|To|Subject|Date|Sent):', line, re.IGNORECASE):
                header_section.append(line)
            elif in_header and line.strip() == '':
                in_header = False
            elif not in_header:
                body_section.append(line)
            else:
                header_section.append(line)
        
        sections['email_header'] = '\n'.join(header_section)
        sections['email_body'] = '\n'.join(body_section)
        
        return sections

    def _map_voucher_sections(self, lines: List[str]) -> Dict[str, str]:
        """Map voucher sections"""
        sections = {'voucher_content': '\n'.join(lines)}
        return sections

    def _map_check_sections(self, lines: List[str]) -> Dict[str, str]:
        """Map check sections"""
        sections = {'check_content': '\n'.join(lines)}
        return sections

    def _semantic_ml_extraction(self, text: str, analysis: Dict) -> Dict[str, Any]:
        """ML extraction with semantic understanding"""
        if self.fallback_mode:
            return {}
        
        try:
            doc_type = analysis['document_type']
            sections = analysis.get('key_sections', {})
            
            # Create business-focused prompt
            prompt = f"""You are a business document analyst. Analyze this {doc_type} document and extract structured business information.

Document Text:
{text[:2500]}

Return ONLY valid JSON with this structure:
{{
  "document_type": "{doc_type}",
  "confidence": 0.0,
  "entities": {{
    "people": [{{"name": "", "role": "", "contact": ""}}],
    "companies": [{{"name": "", "tin": "", "address": ""}}],
    "locations": [{{"address": "", "city": "", "country": ""}}]
  }},
  "key_information": {{
    "reference_numbers": [],
    "amounts": [{{"value": "", "currency": "PHP", "type": ""}}],
    "dates": [],
    "purpose": "",
    "status": ""
  }},
  "communication_data": {{
    "emails": [],
    "phones": [],
    "subject": ""
  }}
}}

Extract ONLY information that actually exists in the document. Do not invent data."""

            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2000, 
                padding=False
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract and validate JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return self._validate_ml_result(result, text)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse ML JSON output")
            
            return {}
            
        except Exception as e:
            logger.error(f"Semantic ML extraction failed: {e}")
            return {}

    def _validate_ml_result(self, result: Dict, source_text: str) -> Dict:
        """Validate ML result against source text"""
        validated = result.copy()
        source_lower = source_text.lower()
        
        # Validate entities
        if 'entities' in validated:
            for entity_type in ['people', 'companies', 'locations']:
                if entity_type in validated['entities']:
                    valid_entities = []
                    for entity in validated['entities'][entity_type]:
                        if isinstance(entity, dict) and 'name' in entity:
                            name = entity['name'].lower()
                            if name and len(name) > 2 and name in source_lower:
                                valid_entities.append(entity)
                    validated['entities'][entity_type] = valid_entities
        
        # Validate key information
        if 'key_information' in validated:
            key_info = validated['key_information']
            for key, value in key_info.items():
                if isinstance(value, str) and value:
                    if value.lower() not in source_lower:
                        key_info[key] = ""
                elif isinstance(value, list):
                    valid_items = []
                    for item in value:
                        if isinstance(item, dict):
                            valid_items.append(item)
                        elif isinstance(item, str) and item.lower() in source_lower:
                            valid_items.append(item)
                    key_info[key] = valid_items
        
        return validated

    def _business_pattern_extraction(self, text: str, analysis: Dict) -> Dict[str, Any]:
        """Business-focused pattern extraction"""
        doc_type = analysis['document_type']
        sections = analysis.get('key_sections', {})
        
        extracted = {
            'entities': {'people': [], 'companies': [], 'locations': []},
            'key_information': {},
            'communication_data': {'emails': [], 'phones': []},
            'amounts': [],
            'dates': []
        }
        
        # Extract companies with context
        for pattern in self.business_patterns['companies']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                company_name = match.group().strip()
                if company_name and len(company_name) > 2:
                    extracted['entities']['companies'].append({
                        'name': company_name,
                        'confidence': 0.9
                    })
        
        # Extract business amounts (not random numbers)
        for pattern in self.business_patterns['amounts']:
            matches = re.finditer(pattern, text)
            for match in matches:
                amount_str = match.group(1) if match.groups() else match.group()
                currency = 'PHP'
                if 'USD' in match.group(0) or '$' in match.group(0):
                    currency = 'USD'
                
                # Validate this is actually an amount, not a random number
                context = text[max(0, match.start()-30):match.end()+30].lower()
                amount_indicators = ['amount', 'total', 'payment', 'php', 'usd', '$', '₱', 'balance']
                
                if any(indicator in context for indicator in amount_indicators):
                    try:
                        amount_val = float(amount_str.replace(',', ''))
                        if amount_val >= 1:  # Meaningful amounts only
                            extracted['amounts'].append({
                                'amount': f"{amount_val:,.2f}",
                                'currency': currency,
                                'context': 'business_transaction'
                            })
                    except ValueError:
                        continue
        
        # Extract proper dates
        for pattern in self.business_patterns['dates']:
            matches = re.findall(pattern, text)
            for match in matches:
                formatted_date = self._format_date(match)
                if formatted_date and formatted_date not in extracted['dates']:
                    extracted['dates'].append(formatted_date)
        
        # Extract real email addresses
        for pattern in self.business_patterns['emails']:
            matches = re.findall(pattern, text)
            for match in matches:
                if '@' in match and '.' in match:
                    extracted['communication_data']['emails'].append(match)
        
        # Extract actual phone numbers (not random digit strings)
        for pattern in self.business_patterns['phones']:
            matches = re.findall(pattern, text)
            for match in matches:
                # Validate it's actually a phone number format
                clean_phone = re.sub(r'[^\d+]', '', match)
                if len(clean_phone) >= 10 and (clean_phone.startswith('+') or len(clean_phone) == 10):
                    extracted['communication_data']['phones'].append(match)
        
        # Extract business reference numbers (not OCR garbage)
        for pattern in self.business_patterns['reference_numbers']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref_num = match.group(1) if match.groups() else match.group()
                # Validate it's a proper reference number
                if len(ref_num) >= 6 and not re.match(r'^[a-z]+$', ref_num.lower()):
                    if 'reference_numbers' not in extracted['key_information']:
                        extracted['key_information']['reference_numbers'] = []
                    extracted['key_information']['reference_numbers'].append(ref_num)
        
        # Extract document-specific business data
        if doc_type == 'payment_request':
            extracted = self._extract_payment_business_data(extracted, sections)
        elif doc_type == 'email':
            extracted = self._extract_email_business_data(extracted, sections)
        
        return extracted

    def _extract_payment_business_data(self, extracted: Dict, sections: Dict) -> Dict:
        """Extract payment-specific business data"""
        # Extract payee from payee section
        if 'payee_section' in sections:
            payee_match = re.search(r'PAYEE[:\s]*([A-Z][A-Z\s&.,]{3,50})', sections['payee_section'], re.IGNORECASE)
            if payee_match:
                extracted['key_information']['payee'] = payee_match.group(1).strip()
        
        # Extract purpose from purpose section
        if 'purpose_section' in sections:
            purpose_match = re.search(r'PURPOSE[\/\s]*DESCRIPTION[:\s]*(.{10,100})', sections['purpose_section'], re.IGNORECASE | re.DOTALL)
            if purpose_match:
                extracted['key_information']['purpose'] = purpose_match.group(1).strip()
        
        # Extract approval info
        if 'approval_section' in sections:
            prepared_match = re.search(r'PREPARED\s*BY[:\s]*([A-Z][A-Z\s\.]{3,30})', sections['approval_section'], re.IGNORECASE)
            approved_match = re.search(r'APPROVED\s*BY[:\s]*([A-Z][A-Z\s\.]{3,30})', sections['approval_section'], re.IGNORECASE)
            
            if prepared_match:
                extracted['entities']['people'].append({
                    'name': prepared_match.group(1).strip(),
                    'role': 'Prepared By'
                })
            if approved_match:
                extracted['entities']['people'].append({
                    'name': approved_match.group(1).strip(),
                    'role': 'Approved By'
                })
        
        return extracted

    def _extract_email_business_data(self, extracted: Dict, sections: Dict) -> Dict:
        """Extract email-specific business data"""
        if 'email_header' in sections:
            header = sections['email_header']
            
            # Extract subject
            subject_match = re.search(r'Subject[:\s]*(.*)', header, re.IGNORECASE)
            if subject_match:
                extracted['communication_data']['subject'] = subject_match.group(1).strip()
        
        return extracted

    def _intelligent_merge(self, ml_data: Dict, pattern_data: Dict, analysis: Dict) -> Dict:
        """Intelligently merge ML and pattern results"""
        merged = pattern_data.copy()
        
        # Merge entities
        if 'entities' in ml_data:
            for entity_type in ['people', 'companies', 'locations']:
                ml_entities = ml_data.get('entities', {}).get(entity_type, [])
                pattern_entities = merged.get('entities', {}).get(entity_type, [])
                
                # Add ML entities that don't exist in pattern results
                for ml_entity in ml_entities:
                    if ml_entity not in pattern_entities:
                        pattern_entities.append(ml_entity)
        
        # Merge key information
        ml_key_info = ml_data.get('key_information', {})
        pattern_key_info = merged.get('key_information', {})
        
        for key, value in ml_key_info.items():
            if key not in pattern_key_info and value:
                pattern_key_info[key] = value
        
        return merged

    def _create_business_result(self, extracted_data: Dict, analysis: Dict, text: str) -> Dict[str, Any]:
        """Create final business-focused result"""
        result = {
            'document_type': analysis['document_type'],
            'confidence': self._calculate_business_confidence(extracted_data, analysis, text),
            'entities': extracted_data.get('entities', {'people': [], 'companies': [], 'locations': []}),
            'key_information': extracted_data.get('key_information', {}),
            'dates_found': extracted_data.get('dates', []),
            'amounts_found': extracted_data.get('amounts', []),
            'processing_confidence': 0.0,
            'extraction_method': f'qwen_{self.model_size}_business_intelligence'
        }
        
        # Add communication data if relevant
        if extracted_data.get('communication_data'):
            comm_data = extracted_data['communication_data']
            if comm_data.get('emails'):
                result['email_addresses'] = comm_data['emails']
            if comm_data.get('phones'):
                result['phone_numbers'] = comm_data['phones']
            if comm_data.get('subject'):
                result['key_information']['subject'] = comm_data['subject']
        
        # Add reference numbers if found
        if extracted_data.get('key_information', {}).get('reference_numbers'):
            result['reference_numbers'] = extracted_data['key_information']['reference_numbers']
        
        result['processing_confidence'] = result['confidence'] * 100
        return result

    def _calculate_business_confidence(self, data: Dict, analysis: Dict, text: str) -> float:
        """Calculate confidence based on business data quality"""
        base_confidence = 0.75
        
        # Document type recognition bonus
        if analysis.get('confidence', 0) > 0.5:
            base_confidence += 0.1
        
        # Entity extraction bonus
        entities = data.get('entities', {})
        entity_count = sum(len(entity_list) for entity_list in entities.values())
        if entity_count > 0:
            base_confidence += min(entity_count * 0.02, 0.1)
        
        # Business data bonus
        key_info = data.get('key_information', {})
        if len(key_info) > 2:
            base_confidence += 0.05
        
        # Communication data bonus
        comm_data = data.get('communication_data', {})
        if comm_data.get('emails') or comm_data.get('phones'):
            base_confidence += 0.02
        
        return min(base_confidence, 0.95)

    def _format_date(self, date_str: str) -> Optional[str]:
        """Format date to YYYY-MM-DD"""
        date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%d-%m-%Y',
            '%m/%d/%y', '%m-%d-%y', '%d/%m/%y', '%d-%m-%y'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                # Handle 2-digit years
                if parsed_date.year < 1950:
                    parsed_date = parsed_date.replace(year=parsed_date.year + 100)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None

    def _create_fallback_result(self, text: str, file_info: Dict = None) -> Dict[str, Any]:
        """Create fallback result for errors"""
        result = {
            'document_type': 'general',
            'confidence': 0.1,
            'entities': {'people': [], 'companies': [], 'locations': []},
            'key_information': {},
            'dates_found': [],
            'amounts_found': [],
            'processing_confidence': 10.0,
            'extraction_method': 'fallback_pattern',
            'processing_time_seconds': 0.1
        }
        
        if file_info:
            result['processing_info'] = {
                'category': 'unknown',
                'classification': 'failed',
                'document_name': file_info.get('document_name', ''),
                'original_filename': file_info.get('original_filename', ''),
                'processed_at': datetime.now().isoformat(),
                'extraction_method': 'fallback_only'
            }
        
        return result

    def __del__(self):
        try:
            if hasattr(self, 'model') and self.model:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer:
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass