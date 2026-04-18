"""
Extraction service for document compliance evaluation.

Provides the evaluate_compliance function that validates extracted invoice
and quotation data against a set of business rules.
"""

import logging
from models import DocumentData

logger = logging.getLogger(__name__)


def evaluate_compliance(data: DocumentData) -> dict:
    """Evaluate if extracted document data meets compliance requirements.

    Rules applied (each failed rule adds a reason and results in Red Light):

    1. Document type must not be 'Unknown'.
    2. Invoices must have a purchase order number (po_number).
    3. Invoices must have bank details (bank_details).
    4. Quotations must have delivery time or terms (delivery_time_or_terms).
    5. All documents must have a document number (document_number).
    6. All documents must have a date of issue (date_of_issue).
    7. All documents must have a total gross amount (total_gross_amount).
    8. When subtotal_amount, tax_amount, and total_gross_amount are all present,
       subtotal_amount + tax_amount must equal total_gross_amount within ±0.05.

    Args:
        data: A DocumentData instance containing the extracted document fields.

    Returns:
        A dictionary with two keys:
        - 'status' (str): 'Green Light' if all rules pass, 'Red Light' otherwise.
        - 'reasons' (list[str]): Explanation for each failed rule (empty on Green Light).

    Example usage::

        # Green Light example
        green_data = DocumentData(
            vendor_name="Acme Corp",
            document_type="Invoice",
            document_number="INV-001",
            date_of_issue="2024-04-15",
            po_number="PO-2024-123",
            bank_details="IBAN: DE89370400440532013000",
            subtotal_amount=1000.0,
            tax_amount=100.0,
            total_gross_amount=1100.0,
        )
        result = evaluate_compliance(green_data)
        # {'status': 'Green Light', 'reasons': []}

        # Red Light example – missing PO number
        red_data = DocumentData(
            vendor_name="Acme Corp",
            document_type="Invoice",
            document_number="INV-001",
            date_of_issue="2024-04-15",
            po_number=None,
            bank_details="IBAN: DE89370400440532013000",
            total_gross_amount=1100.0,
        )
        result = evaluate_compliance(red_data)
        # {'status': 'Red Light', 'reasons': ['Invoice is missing purchase order number']}

        # Red Light example – math error
        math_error_data = DocumentData(
            vendor_name="Acme Corp",
            document_type="Invoice",
            document_number="INV-001",
            date_of_issue="2024-04-15",
            po_number="PO-123",
            bank_details="IBAN: DE89370400440532013000",
            subtotal_amount=1000.0,
            tax_amount=100.0,
            total_gross_amount=1050.0,  # Wrong – should be 1100
        )
        result = evaluate_compliance(math_error_data)
        # {'status': 'Red Light',
        #  'reasons': ['Financial amounts do not add up correctly (subtotal + tax ≠ total)']}
    """
    reasons: list[str] = []

    # 1. Document type validation
    if data.document_type == "Unknown":
        reasons.append("Document type is unknown")

    # 2 & 3. Invoice-specific rules
    if data.document_type == "Invoice":
        if not data.po_number:
            reasons.append("Invoice is missing purchase order number")
        if not data.bank_details:
            reasons.append("Invoice is missing bank details")

    # 4. Quotation-specific rules
    if data.document_type == "Quotation":
        if not data.delivery_time_or_terms:
            reasons.append("Quotation is missing delivery time or terms")

    # 5. Required field: document number
    if not data.document_number:
        reasons.append("Document number is missing")

    # 6. Required field: date of issue
    if not data.date_of_issue:
        reasons.append("Date of issue is missing")

    # 7. Required field: total gross amount
    if data.total_gross_amount is None:
        reasons.append("Total gross amount is missing")

    # 8. Math validation (only when all three amounts are present)
    if (
        data.subtotal_amount is not None
        and data.tax_amount is not None
        and data.total_gross_amount is not None
    ):
        calculated = data.subtotal_amount + data.tax_amount
        if abs(calculated - data.total_gross_amount) > 0.05:
            reasons.append(
                "Financial amounts do not add up correctly (subtotal + tax ≠ total)"
            )

    status = "Green Light" if not reasons else "Red Light"
    logger.debug("evaluate_compliance result: status=%s reasons=%s", status, reasons)
    return {"status": status, "reasons": reasons}
