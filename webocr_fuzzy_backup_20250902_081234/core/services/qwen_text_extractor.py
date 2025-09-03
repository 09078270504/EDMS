# qwen_text_extractor.py
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

################################################################################
# Unified Metadata Schema Prompting
################################################################################

CORE_FIELDS = [
    "doc_type", "title", "ids", "parties", "dates",
    "amounts", "line_items", "payment_terms", "addresses", "contacts",
    "status", "tags", "extracted_entities", "notes", "confidence"
]

TYPE_EXT = {
    "journal_voucher": ["gl"],
    "official_receipt": ["or_details"],
    "collection_receipt": ["or_details"],
    "acknowledgement_receipt": ["or_details"],
    "request_for_payment": ["payment_details"],
    "fund_request": ["payment_details"],
    "petty_cash": ["payment_details"],
    "cash_replenishment": ["payment_details"],
    "delivery_receipt": ["dr_no", "received_by", "received_date"],
    "insurance": ["policy"],
    "policy": ["policy"],
    "ticket": ["travel"],
    "booking_confirmation": ["travel"],
    "payment_details": ["payment_details"],
    "service_report": ["work_details"],
    "service_completion_certificate": ["work_details"],
    "cleaning_services": ["work_details"],
    "operation_record": ["work_details"],
    "site_inspection_witness_charge": ["work_details"],
    "electrical_testing_and_commissioning": ["work_details"],
    "billing_statement": ["period_covered", "previous_balance", "payments", "adjustments"],
    "statement_of_account": ["period_covered", "previous_balance", "payments", "adjustments"],
    "invoice": ["period_covered"],
    "charge_invoice": ["period_covered"],
    "charge_sale_invoice": ["period_covered"],
    "quotation": ["vendor_terms", "delivery_terms"],
    "purchase_order": ["vendor_terms", "delivery_terms"],
    "sales_order": ["vendor_terms", "delivery_terms"],
    "journal_voucher_gl": ["gl"],
}

def _truncate_for_prompt(text: str, limit: int = 6000) -> str:
    if not text:
        return ""
    return (text[:limit] + "...") if len(text) > limit else text

def build_prompt(ocr_text: str, doc_type: str) -> str:
    truncated = _truncate_for_prompt(ocr_text, 6000)
    fields_list = CORE_FIELDS + TYPE_EXT.get(doc_type, [])
    fields_fmt = ", ".join(fields_list)
    return f"""
You are an expert business-document parser. Extract ONLY real values from the OCR text into STRICT JSON.

RULES:
- Use snake_case keys.
- Return EXACTLY these root fields: [{fields_fmt}]
- If a field is not present, set scalars to null and arrays to [].
- "confidence" must be 0.0..1.0.
- Normalize money to numbers (no commas), currency to ISO (e.g., PHP).
- Trim text values; do NOT hallucinate.
- If doc_type is unclear, keep your best guess; do NOT invent fields.

Field guidance:
- ids: document_no, reference_no, po_no, so_no, jo_no, invoice_no, voucher_no, ticket_no
- parties: issuer{{name,tin?}}, recipient{{name,tin?}}, other[]{{role,name}}
- dates: issue_date, due_date, service_period{{from,to}}, coverage{{from,to}}
- amounts: currency, subtotal, tax, total, other_charges[]
- line_items[]: description, qty, uom, unit_price, amount, account?, project?, cost_center?
- extracted_entities: people[], companies[], locations[], projects[]

Per-type (when applicable):
- journal_voucher.gl[]: {{account, debit, credit, party_type, party_name, ref_type, ref_no}}
- *_receipt.or_details: {{or_no, amount_received, payer}}
- *payment*.payment_details: {{method, bank, check_no, gcash_ref}}
- delivery_receipt: dr_no, received_by, received_date
- insurance/policy.policy: {{policy_no, insured, coverage_from, coverage_to, premium}}
- ticket/booking_confirmation.travel: {{airline, booking_ref, passenger, flights[], baggage, fare_basis[]}}
- service_* / operations*.work_details: {{location, scope, serials[], findings, actions, signatories[]}}
- billing_statement/statement_of_account: period_covered, previous_balance, payments, adjustments
- quotation/purchase_order/sales_order: vendor_terms, delivery_terms

DOCUMENT TYPE (best guess allowed): {doc_type}

OCR TEXT:
{truncated}

Return ONLY valid JSON, no explanation.
""".strip()


################################################################################
# Qwen Text Extractor
################################################################################

@dataclass
class TextGenConfig:
    max_new_tokens: int = 800
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05
    stop_strings: Tuple[str, ...] = ("</s>",)
    load_4bit: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class QwenTextExtractor:
    """
    Flexible init:
      - Either pass (model, tokenizer)
      - Or pass model_name=... and it will load for you
      - Any extra kwargs are ignored for compatibility with older code
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        config: Optional[TextGenConfig] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        self.config = config or TextGenConfig()
        self.device = device or self.config.device

        if model_name and (model is None or tokenizer is None):
            quant = None
            if self.config.load_4bit and self.device != "cpu":
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                quantization_config=quant,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                model = model.to("cpu")

        if model is None or tokenizer is None:
            raise ValueError("QwenTextExtractor requires either (model, tokenizer) or model_name.")

        self.model = model
        self.tokenizer = tokenizer

        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=(self.config.temperature > 0),
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    @torch.inference_mode()
    def extract_json(self, ocr_text: str, doc_type: str) -> Dict[str, Any]:
        prompt = build_prompt(ocr_text or "", doc_type or "general")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            generation_config=self.generation_config
        )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        data = self._parse_first_json(text)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _parse_first_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        while start != -1:
            obj, end = QwenTextExtractor._scan_brace_object(text, start)
            if obj is None:
                break
            try:
                return json.loads(obj)
            except Exception:
                start = text.find("{", start + 1)
        return {}

    @staticmethod
    def _scan_brace_object(s: str, start: int) -> Tuple[Optional[str], Optional[int]]:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1], i
        return None, None
