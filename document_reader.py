"""
Document text extraction module for invoices and quotes.

This module provides functionality to extract text from multiple document formats
including PDF, DOCX, and XLSX files. It handles different file types with appropriate
libraries and converts all content to readable string formats suitable for LLM processing.
"""

import os
from pathlib import Path

import pdfplumber
import pandas as pd
from docx import Document
from pdf2image import convert_from_path
import pytesseract


def extract_text_from_file(file_path: str) -> str:
    """Extract text from various document formats (PDF, DOCX, XLSX).

    Determines the file type based on file extension and delegates to the
    appropriate extraction function.

    Args:
        file_path: Path to the document file (PDF, DOCX, or XLSX).

    Returns:
        Extracted text content as a string.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        ValueError: If the file extension is not supported.
        Exception: If there are errors reading or parsing the file content.

    Example:
        >>> text = extract_text_from_file("invoice.pdf")
        >>> print(text[:100])
        'Invoice Number: 12345...'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension == ".docx":
        return extract_text_from_docx(file_path)
    elif extension == ".xlsx":
        return extract_text_from_xlsx(file_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{extension}'. "
            "Supported formats: .pdf, .docx, .xlsx"
        )


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber with OCR fallback.

    Attempts direct text extraction using pdfplumber. If the extracted text
    is less than 50 characters (indicating a scanned/image-based PDF), falls
    back to OCR using pytesseract on images converted from each page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text with page separators. OCR pages are marked accordingly.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: If pdfplumber cannot open the file or pytesseract is not
            installed when OCR fallback is required.

    Example:
        >>> text = extract_text_from_pdf("scanned_invoice.pdf")
        >>> print(text[:50])
        '--- Page 1 (OCR) ---\\nInvoice Number: 12345...'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        text_parts: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")

        text = "\n\n".join(text_parts)
    except Exception as e:
        raise Exception(f"Error reading PDF file '{file_path}': {str(e)}")

    if len(text.strip()) < 50:
        try:
            text = _extract_text_from_pdf_ocr(file_path)
        except Exception as e:
            raise Exception(
                f"OCR extraction failed for '{file_path}'. "
                f"Ensure Tesseract is installed: {str(e)}"
            )

    return text


def _extract_text_from_pdf_ocr(file_path: str) -> str:
    """Extract text from a PDF using OCR (pytesseract).

    Converts each page of the PDF to an image and performs optical character
    recognition using Tesseract.

    Args:
        file_path: Path to the PDF file.

    Returns:
        OCR-extracted text with page separators.

    Raises:
        Exception: If pdf2image or pytesseract encounters an error during
            conversion or recognition.
    """
    try:
        images = convert_from_path(file_path)
        ocr_parts: list[str] = []
        for page_num, image in enumerate(images, start=1):
            page_text = pytesseract.image_to_string(image)
            ocr_parts.append(f"--- Page {page_num} (OCR) ---\n{page_text}")
        return "\n\n".join(ocr_parts)
    except Exception as e:
        raise Exception(f"Error during OCR extraction of '{file_path}': {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file using python-docx.

    Reads all paragraphs from the Word document and joins them with newlines,
    preserving paragraph structure.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Extracted text with paragraph breaks preserved.

    Raises:
        FileNotFoundError: If the DOCX file does not exist.
        Exception: If the DOCX file cannot be read or parsed.

    Example:
        >>> text = extract_text_from_docx("quote.docx")
        >>> print(text[:80])
        'Dear Customer,\\nPlease find attached our quote...'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    try:
        doc = Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        raise Exception(f"Error reading DOCX file '{file_path}': {str(e)}")


def extract_text_from_xlsx(file_path: str) -> str:
    """Extract text from an XLSX file using pandas.

    Reads all sheets from the Excel workbook. For each sheet, drops completely
    empty rows and columns to clean up the data for LLM context windows, then
    converts the cleaned DataFrame to a readable string with a sheet header.

    Args:
        file_path: Path to the XLSX file.

    Returns:
        Extracted data formatted as readable tables with sheet name headers.
        Empty sheets are skipped.

    Raises:
        FileNotFoundError: If the XLSX file does not exist.
        Exception: If the XLSX file cannot be read or parsed.

    Example:
        >>> text = extract_text_from_xlsx("data.xlsx")
        >>> print(text[:80])
        '### Sheet: Invoice\\n Item  Quantity  Price\\n Widget         2  19.99'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XLSX file not found: {file_path}")

    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_contents: list[str] = []

        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)

            # Drop completely empty rows and columns
            df = df.dropna(how="all")
            df = df.dropna(axis=1, how="all")

            if df.empty:
                continue

            sheet_contents.append(f"### Sheet: {sheet_name}")
            sheet_contents.append(df.to_string(index=False))

        return "\n\n".join(sheet_contents)
    except Exception as e:
        raise Exception(f"Error reading XLSX file '{file_path}': {str(e)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_reader.py <file_path>")
        sys.exit(1)

    try:
        extracted = extract_text_from_file(sys.argv[1])
        print(extracted)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)