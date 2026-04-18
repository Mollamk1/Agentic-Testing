"""
Pydantic models for document data extracted from invoices and quotations.
"""

from typing import Optional
from pydantic import BaseModel


class DocumentData(BaseModel):
    """Structured representation of data extracted from an invoice or quotation."""

    vendor_name: Optional[str] = None
    document_type: Optional[str] = None  # 'Invoice', 'Quotation', or 'Unknown'
    document_number: Optional[str] = None
    date_of_issue: Optional[str] = None
    po_number: Optional[str] = None
    bank_details: Optional[str] = None
    delivery_time_or_terms: Optional[str] = None
    subtotal_amount: Optional[float] = None
    tax_amount: Optional[float] = None
    total_gross_amount: Optional[float] = None
