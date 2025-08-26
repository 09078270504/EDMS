import re
import logging
from typing import Dict, List, Tuple

class DocumentFormatter:
    """Formats raw OCR text into structured, readable documents"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_document(self, raw_text: str, document_type: str = "general") -> str:
        """Format raw OCR text into structured document"""
        try:
            self.logger.info(f"Formatting {document_type} document")
            
            # Clean and normalize the text first
            cleaned_text = self._normalize_text(raw_text)
            
            # Detect specific document type if not provided
            if document_type == "general":
                document_type = self._detect_document_type(cleaned_text)
            
            # Apply specific formatting based on document type
            if "check" in document_type.lower() or "voucher" in document_type.lower():
                return self._format_check_voucher(cleaned_text)
            elif "payment" in document_type.lower() or "request" in document_type.lower():
                return self._format_payment_request(cleaned_text)
            elif "receipt" in document_type.lower():
                return self._format_receipt(cleaned_text)
            else:
                return self._format_general_document(cleaned_text)
                
        except Exception as e:
            self.logger.error(f"Document formatting failed: {e}")
            return raw_text  # Return original if formatting fails
    
    def _normalize_text(self, text: str) -> str:
        """Clean and normalize raw OCR text"""
        # Remove page headers
        text = re.sub(r'--- Page \d+ ---\s*', '', text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type from content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['check voucher', 'voucher', 'batch number']):
            return 'check_voucher'
        elif any(word in text_lower for word in ['request for payment', 'payment request']):
            return 'payment_request'
        elif any(word in text_lower for word in ['receipt', 'received from']):
            return 'receipt'
        else:
            return 'general'
    
    def _format_check_voucher(self, text: str) -> str:
        """Format check voucher documents"""
        formatted = "--- Page 1 ---\n"
        
        # Extract company information
        company_info = self._extract_company_info(text)
        if company_info:
            formatted += f"{company_info}\n\n"
        
        # Document type
        formatted += "Check Voucher\n\n"
        
        # Extract key fields
        fields = self._extract_voucher_fields(text)
        
        # Format main information
        for field_name, value in fields.items():
            if value:
                formatted += f"{field_name}: {value}\n"
        
        formatted += "\n"
        
        # Extract table data if present
        table_data = self._extract_table_data(text)
        if table_data:
            formatted += self._format_table(table_data) + "\n\n"
        
        # Extract received information
        received_info = self._extract_received_info(text)
        if received_info:
            formatted += f"{received_info}\n\n"
        
        # Add signature section
        formatted += self._format_signature_section(text)
        
        return formatted
    
    def _format_payment_request(self, text: str) -> str:
        """Format payment request documents"""
        formatted = "--- Page 1 ---\n"
        
        # Extract company header
        company_info = self._extract_company_info(text)
        if company_info:
            formatted += f"{company_info}\n\n"
        
        # Document type
        formatted += "REQUEST FOR PAYMENT\n\n"
        
        # Extract and format key information
        info_fields = self._extract_payment_request_fields(text)
        
        for section, fields in info_fields.items():
            if fields:
                formatted += f"{section.upper()}:\n"
                for field, value in fields.items():
                    if value:
                        formatted += f"  {field}: {value}\n"
                formatted += "\n"
        
        # Extract amounts and create summary
        amounts = self._extract_amounts(text)
        if amounts:
            formatted += "AMOUNT BREAKDOWN:\n"
            for amount_info in amounts:
                formatted += f"  {amount_info}\n"
            formatted += "\n"
        
        # Add signature section
        formatted += self._format_signature_section(text)
        
        return formatted
    
    def _format_receipt(self, text: str) -> str:
        """Format receipt documents"""
        formatted = "--- Page 1 ---\n"
        
        # Extract company info
        company_info = self._extract_company_info(text)
        if company_info:
            formatted += f"{company_info}\n\n"
        
        formatted += "RECEIPT\n\n"
        
        # Extract receipt details
        receipt_fields = self._extract_receipt_fields(text)
        for field, value in receipt_fields.items():
            if value:
                formatted += f"{field}: {value}\n"
        
        return formatted
    
    def _format_general_document(self, text: str) -> str:
        """Format general documents with basic structure"""
        formatted = "--- Page 1 ---\n"
        
        # Try to identify and format key sections
        sections = self._identify_sections(text)
        
        for section_name, content in sections.items():
            if content:
                formatted += f"{section_name.upper()}:\n{content}\n\n"
        
        return formatted
    
    def _extract_company_info(self, text: str) -> str:
        """Extract company name and address"""
        # Look for company patterns
        company_patterns = [
            r'(Comfac Corporation|EEI Corporation|nfac Global Group)',
            r'(\d+\s+[A-Za-z\s]+Street[^,]*,?\s*[A-Za-z\s]+)',
        ]
        
        company_info = []
        for pattern in company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                if match and match not in company_info:
                    company_info.append(match.strip())
        
        return '\n'.join(company_info) if company_info else ""
    
    def _extract_voucher_fields(self, text: str) -> Dict[str, str]:
        """Extract check voucher specific fields"""
        fields = {}
        
        # Define field patterns
        patterns = {
            'Batch Number': r'Batch Number\s*(\d+)',
            'Paid to': r'Paid to\s*(\d+\s*[A-Z\s-]+)',
            'Account Code': r'Account Code\s*(\d+\s*[A-Z\s-]+)',
            'Total Amount': r'Total Amount\s*([\d,]+\.?\d*)',
            'Check Number': r'Check Number\s*([A-Z0-9#]+)',
        }
        
        for field_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields[field_name] = match.group(1).strip()
        
        return fields
    
    def _extract_payment_request_fields(self, text: str) -> Dict[str, Dict[str, str]]:
        """Extract payment request fields organized by section"""
        sections = {
            'company_info': {},
            'payee_info': {},
            'payment_details': {}
        }
        
        # Company section
        if 'COMPANY' in text:
            company_match = re.search(r'COMPANY\s+([^P]+?)(?:PAYEE|$)', text, re.IGNORECASE)
            if company_match:
                sections['company_info']['Name'] = company_match.group(1).strip()
        
        # Payee section
        if 'PAYEE' in text:
            payee_match = re.search(r'PAYEE\s+([^D]+?)(?:DATE|$)', text, re.IGNORECASE)
            if payee_match:
                sections['payee_info']['Name'] = payee_match.group(1).strip()
        
        # Date
        date_match = re.search(r'DATE:\s*([\d-/]+)', text, re.IGNORECASE)
        if date_match:
            sections['payment_details']['Date'] = date_match.group(1)
        
        # Check number
        check_match = re.search(r'CHECK\s*NO\.?\s*(\d+)', text, re.IGNORECASE)
        if check_match:
            sections['payment_details']['Check Number'] = check_match.group(1)
        
        return sections
    
    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts"""
        amounts = []
        
        # Pattern for amounts
        amount_patterns = [
            r'\$\s*([\d,]+\.?\d*)',
            r'PHP\s*([\d,]+\.?\d*)',
            r'₱\s*([\d,]+\.?\d*)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3:  # Avoid single digits
                    amounts.append(f"Amount: {match}")
        
        return amounts[:3]  # Limit to first 3 amounts
    
    def _extract_table_data(self, text: str) -> List[List[str]]:
        """Extract tabular data"""
        # Look for patterns that suggest table structure
        # This is a simplified approach - could be enhanced
        
        if 'V.P. No.' in text or 'INV. No.' in text:
            # Sample table structure for vouchers
            return [
                ['V.P. No.', 'INV. No.', 'Applied'],
                ['', '', '']  # Placeholder for actual data
            ]
        
        return []
    
    def _format_table(self, table_data: List[List[str]]) -> str:
        """Format table with proper alignment"""
        if not table_data:
            return ""
        
        # Calculate column widths
        col_widths = []
        for col_idx in range(len(table_data[0])):
            max_width = max(len(str(row[col_idx])) for row in table_data)
            col_widths.append(max(max_width, 10))  # Minimum width of 10
        
        # Format table
        formatted_table = ""
        
        # Header row
        header_row = "|"
        separator_row = "|"
        for i, header in enumerate(table_data[0]):
            header_row += f" {header:<{col_widths[i]}} |"
            separator_row += f"{'-' * (col_widths[i] + 2)}|"
        
        formatted_table += header_row + "\n" + separator_row + "\n"
        
        # Data rows
        for row in table_data[1:]:
            data_row = "|"
            for i, cell in enumerate(row):
                data_row += f" {str(cell):<{col_widths[i]}} |"
            formatted_table += data_row + "\n"
        
        return formatted_table
    
    def _extract_received_info(self, text: str) -> str:
        """Extract received/payment information"""
        received_patterns = [
            r'Received from[^.]+\.',
            r'in the Amount of[^.]+\.',
            r'in payment of[^.]+\.'
        ]
        
        received_info = []
        for pattern in received_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            received_info.extend(matches)
        
        return '\n'.join(received_info) if received_info else ""
    
    def _extract_receipt_fields(self, text: str) -> Dict[str, str]:
        """Extract receipt specific fields"""
        fields = {}
        
        patterns = {
            'Receipt Number': r'Receipt\s*(?:No\.?|Number)\s*(\d+)',
            'Date': r'Date\s*:?\s*([\d/-]+)',
            'Amount': r'Amount\s*:?\s*([\d,]+\.?\d*)',
            'From': r'From\s*:?\s*([^:]+?)(?:To|Date|Amount|$)',
        }
        
        for field_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields[field_name] = match.group(1).strip()
        
        return fields
    
    def _format_signature_section(self, text: str) -> str:
        """Format signature and approval section"""
        signature_section = "\n_________ A P P R O V E D __________\n"
        
        # Look for signature fields
        signature_fields = ['Prepared By', 'Audited By', 'Accounting', 'Treasury', 'Approved By', 'Received By']
        
        found_fields = []
        for field in signature_fields:
            if field.lower() in text.lower():
                found_fields.append(field)
        
        if found_fields:
            signature_section += "  ".join([f"{field}: __________" for field in found_fields[:4]]) + "\n\n"
            
            if len(found_fields) > 4:
                signature_section += "  ".join([f"{field}: __________" for field in found_fields[4:]]) + "\n\n"
        else:
            signature_section += "Prepared By: __________ Audited By: __________ Accounting __________ Treasury\n\n"
            signature_section += "Received By: __________ Date: __________\n"
        
        return signature_section
    
    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract general document sections"""
        sections = {}
        
        # Look for common section markers
        section_markers = [
            'PURPOSE', 'DESCRIPTION', 'AMOUNT', 'DATE', 'COMPANY', 'PAYEE',
            'PREPARED BY', 'APPROVED BY', 'CERTIFICATION'
        ]
        
        current_section = 'HEADER'
        current_content = []
        
        words = text.split()
        
        for word in words:
            # Check if this word starts a new section
            found_marker = None
            for marker in section_markers:
                if word.upper().startswith(marker):
                    found_marker = marker
                    break
            
            if found_marker:
                # Save previous section
                if current_content:
                    sections[current_section] = ' '.join(current_content).strip()
                
                # Start new section
                current_section = found_marker
                current_content = [word]
            else:
                current_content.append(word)
        
        # Save last section
        if current_content:
            sections[current_section] = ' '.join(current_content).strip()
        
        return sections