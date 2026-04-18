"""
Pydantic models for structured LLM output extraction from invoices and quotations.

DocumentData is used to validate and serialize structured data extracted by an LLM
from invoice or quotation documents. Each field mirrors a common field found on such
documents, with descriptive metadata so that the LLM understands exactly what value
to place in each slot.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentData(BaseModel):
    """
    Structured representation of data extracted from an invoice or quotation document.

    This model is intended to be used as the response schema for LLM structured-output
    extraction.  Pass it (or its JSON schema) to the LLM as the required output format
    so that all extracted fields are validated and typed before being consumed by the
    application.
    """

    vendor_name: str = Field(
        ...,
        description=(
            "The full legal name of the vendor or company issuing the document, "
            "e.g. 'Acme Corporation' or 'Smith & Sons Ltd'."
        ),
        examples=["Acme Corporation", "Smith & Sons Ltd"],
    )

    vendor_id_or_tax_id: Optional[str] = Field(
        default=None,
        description=(
            "Tax ID, VAT registration number, EIN, or other business registration "
            "identifier printed on the document."
        ),
        examples=["VAT123456789", "EIN12-3456789", "GB123 4567 89"],
    )

    document_type: Literal["Invoice", "Quotation", "Unknown"] = Field(
        ...,
        description=(
            "The type of document being processed. Use 'Invoice' for bills/invoices, "
            "'Quotation' for quotes/estimates, and 'Unknown' when the type cannot be "
            "determined."
        ),
        examples=["Invoice", "Quotation", "Unknown"],
    )

    document_number: Optional[str] = Field(
        default=None,
        description=(
            "The unique reference number of the document such as an invoice number, "
            "quotation number, or other document identifier."
        ),
        examples=["INV-2024-001", "QT-12345", "REF-2024-0099"],
    )

    date_of_issue: Optional[str] = Field(
        default=None,
        description=(
            "The date the document was issued, formatted as YYYY-MM-DD (ISO 8601). "
            "Convert any other date format found on the document to this format."
        ),
        examples=["2024-01-15", "2024-06-30"],
    )

    po_number: Optional[str] = Field(
        default=None,
        description=(
            "Purchase order number referenced on the document, if present."
        ),
        examples=["PO-2024-5678", "4500012345"],
    )

    currency: Optional[str] = Field(
        default=None,
        description=(
            "Three-letter ISO 4217 currency code for all monetary amounts on the "
            "document."
        ),
        examples=["USD", "EUR", "GBP", "JPY"],
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )

    subtotal_amount: Optional[float] = Field(
        default=None,
        description=(
            "The subtotal amount before taxes or additional fees are applied. "
            "Use the same currency as indicated by the 'currency' field."
        ),
        examples=[1000.00, 2500.50],
        ge=0,
    )

    tax_amount: Optional[float] = Field(
        default=None,
        description=(
            "The total tax or VAT amount shown on the document. "
            "Use the same currency as indicated by the 'currency' field."
        ),
        examples=[200.00, 475.10],
        ge=0,
    )

    total_gross_amount: Optional[float] = Field(
        default=None,
        description=(
            "The final total amount payable, including all taxes and fees. "
            "Use the same currency as indicated by the 'currency' field."
        ),
        examples=[1200.00, 2975.60],
        ge=0,
    )

    payment_terms: Optional[str] = Field(
        default=None,
        description=(
            "Payment terms and conditions as stated on the document."
        ),
        examples=["Net 30", "Due on receipt", "2/10 Net 30", "50% upfront, 50% on delivery"],
    )

    bank_details: Optional[str] = Field(
        default=None,
        description=(
            "Bank account information provided for payment, including details such as "
            "bank name, account number, IBAN, SWIFT/BIC, or routing number."
        ),
        examples=[
            "Bank: HSBC | IBAN: GB29 NWBK 6016 1331 9268 19 | BIC: MIDLGB22",
            "Routing: 021000021 | Account: 123456789",
        ],
    )

    billing_and_shipping_address: Optional[str] = Field(
        default=None,
        description=(
            "The full billing and/or shipping address as printed on the document. "
            "Include both addresses if they differ, separated by a newline."
        ),
        examples=[
            "123 Main St, Springfield, IL 62701, USA",
            "Billing: 10 Downing St, London SW1A 2AA, UK\nShipping: same",
        ],
    )

    delivery_time_or_terms: Optional[str] = Field(
        default=None,
        description=(
            "Delivery schedule, lead time, or shipping terms stated on the document."
        ),
        examples=["Ships within 5–7 business days", "FOB Destination", "Ex Works (EXW)"],
    )

    model_config = {
        "json_schema_extra": {
            "title": "DocumentData",
            "description": (
                "Structured data extracted from an invoice or quotation document "
                "by an LLM."
            ),
        }
    }
