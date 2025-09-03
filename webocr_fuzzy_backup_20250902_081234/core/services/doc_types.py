# -*- coding: utf-8 -*-
"""
Shared registry of document types: synonyms/keywords/expected fields.
Import DOC_TYPES from this module anywhere you need type detection or scoring.
"""

DOC_TYPES = {
    "purchase_order": {
        "synonyms": ["po", "purchase order", "p.o.", "acceptance of purchase order"],
        "keywords": ["purchase order", "po no", "supplier", "delivery", "unit price", "amount"],
        "expected_fields": ["po_number", "supplier_name", "date", "delivery_address", "total_amount", "items"]
    },
    "sales_invoice": {
        "synonyms": ["sales invoice", "charge sale invoice", "invoice dsa", "billing invoice", "invoice"],
        "keywords": ["invoice", "sales invoice", "si no", "amount due", "vat", "tin"],
        "expected_fields": ["invoice_number", "customer_name", "date", "total_amount", "items"]
    },
    "official_receipt": {
        "synonyms": ["official receipt", "collection receipt", "acknowledgement receipt", "teller’s copy", "or"],
        "keywords": ["official receipt", "o.r.", "received from", "or no", "amount received"],
        "expected_fields": ["receipt_number", "payer_name", "date", "total_amount"]
    },
    "credit_memo": {
        "synonyms": ["credit memo", "credit note", "cm", "ar cm"],
        "keywords": ["credit memo", "cm no", "ar cm", "vat", "bir permit", "invoice"],
        "expected_fields": ["ar_cm_no", "date", "invoice_number", "period", "vat_sales", "total"]
    },
    "request_for_payment": {
        "synonyms": ["request for payment", "fund request", "payment request", "summary of payment"],
        "keywords": ["request for payment", "fund request", "payee", "amount", "purpose"],
        "expected_fields": ["rfp_number", "payee", "date", "purpose", "amount"]
    },
    "voucher": {
        "synonyms": ["payable voucher", "check voucher", "voucher"],
        "keywords": ["voucher", "batch number", "check number", "paid to"],
        "expected_fields": ["voucher_number", "payee", "date", "amount", "check_number"]
    },
    "journal_voucher": {
        "synonyms": ["journal voucher", "jv"],
        "keywords": ["journal voucher", "debit", "credit", "account code"],
        "expected_fields": ["jv_number", "date", "lines", "total_debit", "total_credit"]
    },
    "delivery_receipt": {
        "synonyms": ["delivery receipt", "dr", "transmittal"],
        "keywords": ["delivery receipt", "dr no", "delivered to", "qty"],
        "expected_fields": ["dr_number", "date", "consignee", "items"]
    },
    "statement": {
        "synonyms": ["statement of account", "billing statement", "accounting", "soa"],
        "keywords": ["statement of account", "soa", "previous balance", "current charges", "amount due"],
        "expected_fields": ["statement_number", "period", "customer_name", "total_amount"]
    },
    "quotation": {
        "synonyms": ["quotation", "quote"],
        "keywords": ["quotation", "quote no", "validity", "price"],
        "expected_fields": ["quote_number", "date", "customer_name", "items", "total_amount"]
    },
    "tax_certificate": {
        "synonyms": ["certificate of credible tax", "certificate of tax withheld at source", "cedula"],
        "keywords": ["withholding", "2307", "tax", "certificate"],
        "expected_fields": ["certificate_number", "period", "taxpayer", "amount"]
    },
    "email": {
        "synonyms": ["email"],
        "keywords": ["from:", "to:", "subject:", "sent:"],
        "expected_fields": ["sender_email", "subject", "date_sent"]
    },
    "expense_liquidation": {
        "synonyms": ["cash replenishment", "petty cash", "liquidation", "presentation expenses"],
        "keywords": ["petty cash", "liquidation", "replenishment", "expense"],
        "expected_fields": ["report_number", "employee_name", "date", "items", "total_amount"]
    },
    "service_report": {
        "synonyms": ["service report", "service completion certificate", "operation record", "cleaning services", "electrical testing and commissioning"],
        "keywords": ["service report", "date", "site", "findings", "work done"],
        "expected_fields": ["report_number", "date", "client", "site", "work_done"]
    },
    "policies": {
        "synonyms": ["casualty policy", "engineering policy", "insurance"],
        "keywords": ["policy", "insured", "coverage", "premium"],
        "expected_fields": ["policy_number", "insured", "period", "coverage"]
    },
    "freight_billing": {
        "synonyms": ["outbound freight billing", "inbound freight invoice"],
        "keywords": ["freight", "waybill", "awb", "billing"],
        "expected_fields": ["waybill_number", "date", "shipper", "consignee", "amount"]
    },
    "bank_payments": {
        "synonyms": ["gcash transfer receipt", "credit card", "payment details", "universal transaction receipt", "electronic ticket receipt", "airbnb receipt", "booking confirmation"],
        "keywords": ["transaction id", "amount", "date", "card", "gcash", "reference"],
        "expected_fields": ["reference_number", "date", "amount", "payer", "merchant"]
    },
    "reports": {
        "synonyms": ["dashboard", "monthly report", "sales order report", "schedule", "ccg gps tracker"],
        "keywords": ["report", "summary", "dashboard", "schedule"],
        "expected_fields": ["report_title", "period", "prepared_by"]
    },
    "requests": {
        "synonyms": ["request for action", "request for financial assistance"],
        "keywords": ["request", "purpose", "amount"],
        "expected_fields": ["request_number", "requestor", "date", "purpose", "amount"]
    },
    "bank_statement": {
        "synonyms": ["bank statement", "account statement"],
        "keywords": ["bank statement", "statement period", "balance", "transaction"],
        "expected_fields": ["account_number", "period", "ending_balance"]
    },
    "general": {
        "synonyms": [],
        "keywords": [],
        "expected_fields": ["document_number", "date", "amount"]
    }
}
