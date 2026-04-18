"""
Pydantic models for structured document data extraction.

DocumentData represents the structured fields that can be extracted from
invoice and quotation documents.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """A single line item in an invoice or quotation."""

    description: Optional[str] = Field(
        default=None, description="Description of the product or service"
    )
    quantity: Optional[float] = Field(
        default=None, description="Quantity of the item"
    )
    unit_price: Optional[float] = Field(
        default=None, description="Price per unit"
    )
    total_price: Optional[float] = Field(
        default=None, description="Total price for this line item"
    )
    unit: Optional[str] = Field(
        default=None, description="Unit of measurement (e.g. pcs, kg, hours)"
    )


class DocumentData(BaseModel):
    """
    Structured data extracted from an invoice or quotation document.

    All fields are optional. If a field cannot be found or extracted with
    confidence from the source text, it will be None rather than guessed.
    """

    # ---------- Document identity ------------------------------------------ #
    document_type: Optional[str] = Field(
        default=None,
        description="Type of document: 'invoice', 'quotation', 'purchase_order', etc.",
    )
    document_number: Optional[str] = Field(
        default=None,
        description="Invoice / quotation / PO number (e.g. INV-2024-001)",
    )

    # ---------- Dates ------------------------------------------------------- #
    document_date: Optional[str] = Field(
        default=None,
        description="Date the document was issued, formatted as YYYY-MM-DD",
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Payment due date, formatted as YYYY-MM-DD",
    )
    delivery_date: Optional[str] = Field(
        default=None,
        description="Expected delivery date, formatted as YYYY-MM-DD",
    )

    # ---------- Vendor / Seller --------------------------------------------- #
    vendor_name: Optional[str] = Field(
        default=None, description="Name of the selling company or individual"
    )
    vendor_tax_id: Optional[str] = Field(
        default=None, description="VAT / Tax ID of the vendor"
    )
    vendor_address: Optional[str] = Field(
        default=None, description="Full address of the vendor"
    )
    vendor_email: Optional[str] = Field(
        default=None, description="Contact email of the vendor"
    )
    vendor_phone: Optional[str] = Field(
        default=None, description="Contact phone number of the vendor"
    )

    # ---------- Buyer / Customer -------------------------------------------- #
    buyer_name: Optional[str] = Field(
        default=None, description="Name of the buying company or individual"
    )
    buyer_tax_id: Optional[str] = Field(
        default=None, description="VAT / Tax ID of the buyer"
    )
    buyer_address: Optional[str] = Field(
        default=None, description="Full address of the buyer"
    )

    # ---------- Financial summary ------------------------------------------- #
    currency: Optional[str] = Field(
        default=None,
        description="3-letter ISO 4217 currency code (e.g. USD, EUR, GBP)",
    )
    subtotal: Optional[float] = Field(
        default=None, description="Subtotal before tax and discounts"
    )
    tax_amount: Optional[float] = Field(
        default=None, description="Total tax / VAT amount"
    )
    tax_rate: Optional[float] = Field(
        default=None, description="Tax / VAT rate as a percentage (e.g. 20.0 for 20%)"
    )
    discount_amount: Optional[float] = Field(
        default=None, description="Total discount amount"
    )
    total_amount: Optional[float] = Field(
        default=None, description="Grand total amount due"
    )

    # ---------- Payment ----------------------------------------------------- #
    payment_terms: Optional[str] = Field(
        default=None,
        description="Payment terms (e.g. 'Net 30', 'Due on receipt')",
    )
    payment_method: Optional[str] = Field(
        default=None, description="Accepted or used payment method"
    )
    bank_account: Optional[str] = Field(
        default=None, description="Bank account / IBAN for payment"
    )

    # ---------- Line items -------------------------------------------------- #
    line_items: Optional[List[LineItem]] = Field(
        default=None, description="List of individual line items in the document"
    )

    # ---------- Miscellaneous ----------------------------------------------- #
    notes: Optional[str] = Field(
        default=None, description="Additional notes or remarks on the document"
    )
    purchase_order_number: Optional[str] = Field(
        default=None, description="Referenced purchase order number"
    )
