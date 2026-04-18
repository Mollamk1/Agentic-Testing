"""
Extraction and compliance service for invoice and quotation documents.

Provides two main functions:

- ``extract_data_from_text(raw_text)`` – Uses the OpenAI structured-output API
  (``client.beta.chat.completions.parse``) to convert raw document text into a
  validated :class:`~models.DocumentData` Pydantic model.

- ``evaluate_compliance(data)`` – Validates a :class:`~models.DocumentData`
  instance against a set of business rules and returns a Green Light / Red Light
  compliance result.

Example usage::

    from extraction_service import extract_data_from_text, evaluate_compliance

    text = \"\"\"
    INVOICE #INV-001
    Date: 2024-04-15
    Vendor: Acme Corp (Tax ID: 99-12345)
    PO Number: PO-445566
    Subtotal: $100.00
    Tax: $10.00
    Total: $110.00 USD
    Bank Account: 123456789
    \"\"\"

    data = extract_data_from_text(text)
    result = evaluate_compliance(data)
    print(result)  # {'status': 'Green Light', 'reasons': []}
"""

import logging
import os
from typing import Optional

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from models import DocumentData

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Default model – gpt-4o-mini gives a good balance of cost and accuracy.
# Override by setting the OPENAI_MODEL environment variable.
_DEFAULT_MODEL = "gpt-4o-mini"

# System prompt instructs the model to act as a strict Procurement and Finance
# reviewer. It must return null/None for any field not explicitly present in
# the document rather than guessing.
_SYSTEM_PROMPT = (
    "You are a strict Procurement and Finance document reviewer. "
    "Your task is to extract structured data from raw invoice or quotation text "
    "with maximum precision and accuracy.\n\n"
    "Rules:\n"
    "- Extract ONLY information that is explicitly present in the document.\n"
    "- If a field is not clearly present, return null – DO NOT guess or infer.\n"
    "- Accuracy is more important than completeness.\n"
    "- For document_type use 'Invoice' for bills/invoices, 'Quotation' for "
    "quotes/estimates, and 'Unknown' when it cannot be determined.\n"
    "- Format all dates as YYYY-MM-DD (e.g. 2024-04-15).\n"
    "- Express currency as a 3-letter ISO 4217 code (e.g. USD, EUR, GBP). "
    "Derive the code from symbols ($→USD, €→EUR, £→GBP) only when the context "
    "makes the currency unambiguous.\n"
    "- Return numeric amounts as plain decimal numbers without currency symbols "
    "or thousand-separator commas (e.g. 1000.00, not $1,000.00).\n"
    "- For bank_details, include all available payment information: bank name, "
    "account number, IBAN, SWIFT/BIC, or routing number.\n"
)


# --------------------------------------------------------------------------- #
# Extraction Function
# --------------------------------------------------------------------------- #


def extract_data_from_text(
    raw_text: str,
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> DocumentData:
    """Extract structured invoice/quotation data from raw text using OpenAI.

    Sends the provided text to the OpenAI API using the structured-output
    feature (``client.beta.chat.completions.parse``) and returns a validated
    :class:`~models.DocumentData` instance.  Any field not explicitly present
    in the source text will be ``None``.

    Args:
        raw_text: The raw text content of an invoice or quotation document.
        temperature: Sampling temperature (default 0.0 for deterministic output).
        model: OpenAI model name to use.  Defaults to the ``OPENAI_MODEL``
               environment variable, or ``gpt-4o-mini`` if not set.

    Returns:
        A :class:`~models.DocumentData` instance with all extractable fields
        populated.  Fields not found in the text are set to ``None``.

    Raises:
        ValueError: If ``raw_text`` is empty or the ``OPENAI_API_KEY``
                    environment variable is not set.
        AuthenticationError: If the OpenAI API key is invalid or revoked.
        APIConnectionError: If the network connection to the OpenAI API fails.
        RateLimitError: If the request is rejected due to rate limiting.
        APIError: For any other OpenAI API-level error.

    Example::

        text = \"\"\"
        INVOICE #INV-2024-001
        Vendor: Acme Corp
        Date: 2024-04-15
        Total: $1,100.00 USD
        \"\"\"
        data = extract_data_from_text(text)
        print(data.vendor_name)          # 'Acme Corp'
        print(data.total_gross_amount)   # 1100.0
    """
    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must be a non-empty string.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. "
            "Set the OPENAI_API_KEY environment variable."
        )

    selected_model = model or os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)

    # ------------------------------------------------------------------ #
    # Call the OpenAI structured-output API
    # ------------------------------------------------------------------ #
    client = OpenAI(api_key=api_key)

    user_message = (
        "Please extract and structure all available information from the "
        "following document text:\n\n"
        f"{raw_text}"
    )

    logger.info(
        "Sending %d characters to OpenAI model '%s' for extraction.",
        len(raw_text),
        selected_model,
    )

    try:
        response = client.beta.chat.completions.parse(
            model=selected_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=DocumentData,
            temperature=temperature,
        )

        document_data: DocumentData = response.choices[0].message.parsed

        logger.info(
            "Extraction successful. vendor_name=%r, document_type=%r, "
            "total_gross_amount=%r",
            document_data.vendor_name,
            document_data.document_type,
            document_data.total_gross_amount,
        )
        return document_data

    except AuthenticationError:
        logger.error(
            "OpenAI authentication failed. Check that OPENAI_API_KEY is valid."
        )
        raise

    except APIConnectionError:
        logger.error(
            "Could not connect to the OpenAI API. Check your network connection."
        )
        raise

    except RateLimitError:
        logger.error("OpenAI rate limit exceeded. Wait before retrying.")
        raise

    except APIError as exc:
        logger.error("OpenAI API error: %s", exc)
        raise


# --------------------------------------------------------------------------- #
# Compliance Evaluation Function
# --------------------------------------------------------------------------- #


def evaluate_compliance(data: DocumentData) -> dict:
    """Evaluate whether extracted document data meets compliance requirements.

    Rules applied (each failed rule contributes a reason and results in a
    Red Light):

    1. ``document_type`` must not be ``'Unknown'``.
    2. Invoices must have a ``po_number`` (purchase order number).
    3. Invoices must have ``bank_details`` (payment information).
    4. Quotations must have ``delivery_time_or_terms``.
    5. All documents must have a ``document_number``.
    6. All documents must have a ``date_of_issue``.
    7. All documents must have a ``total_gross_amount``.
    8. When ``subtotal_amount``, ``tax_amount``, and ``total_gross_amount`` are
       all present: ``subtotal + tax`` must equal ``total`` within ±0.05.

    Args:
        data: A :class:`~models.DocumentData` instance with the extracted fields.

    Returns:
        A dictionary with two keys:

        - ``'status'`` (``str``): ``'Green Light'`` if all rules pass,
          ``'Red Light'`` otherwise.
        - ``'reasons'`` (``list[str]``): Human-readable explanation for each
          failed rule.  Empty list when status is ``'Green Light'``.

    Example::

        # Green Light
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

        # Red Light – missing PO number
        red_data = DocumentData(
            vendor_name="Acme Corp",
            document_type="Invoice",
            document_number="INV-002",
            date_of_issue="2024-04-15",
            bank_details="IBAN: DE89370400440532013000",
            total_gross_amount=500.0,
        )
        result = evaluate_compliance(red_data)
        # {'status': 'Red Light',
        #  'reasons': ['Invoice is missing purchase order number']}
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
    logger.debug(
        "evaluate_compliance result: status=%s reasons=%s", status, reasons
    )
    return {"status": status, "reasons": reasons}


# --------------------------------------------------------------------------- #
# Test / Demo section
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    fake_extracted_text = """
    INVOICE #99823
    Date: 2023-10-15
    Vendor: Acme Corp (Tax ID: 99-12345)
    PO Number: PO-445566
    Subtotal: $100.00
    Tax: $10.00
    Total: $110.00 USD
    Please pay to Bank Account 123456789.
    """

    print("1. Extracting data with AI...")
    ai_data = extract_data_from_text(fake_extracted_text)
    print(ai_data.model_dump_json(indent=2))

    print("\n2. Running Compliance Check...")
    result = evaluate_compliance(ai_data)
    print(f"Status: {result['status']}")
    print(f"Reasons: {result['reasons']}")

    print("\n--- Green Light scenario ---")
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
    green_result = evaluate_compliance(green_data)
    print(f"Status: {green_result['status']}")
    print(f"Reasons: {green_result['reasons']}")

    print("\n--- Red Light scenario (math error + missing PO) ---")
    red_data = DocumentData(
        vendor_name="Bad Vendor Inc",
        document_type="Invoice",
        document_number="INV-999",
        date_of_issue="2024-04-15",
        bank_details="Account: 987654321",
        subtotal_amount=500.0,
        tax_amount=50.0,
        total_gross_amount=600.0,  # Wrong: should be 550.0
    )
    red_result = evaluate_compliance(red_data)
    print(f"Status: {red_result['status']}")
    print(f"Reasons: {red_result['reasons']}")
