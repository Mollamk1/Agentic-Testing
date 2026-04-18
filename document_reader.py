import os
from typing import Optional

import pdfplumber
import pandas as pd
from docx import Document
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a file based on its extension.

    Args:
        file_path (str): The path to the file (PDF, DOCX, or XLSX).

    Returns:
        str: The extracted text from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = file_path.split('.')[-1].lower()

    if extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif extension == 'docx':
        return extract_text_from_docx(file_path)
    elif extension == 'xlsx':
        return extract_text_from_xlsx(file_path)
    else:
        raise ValueError("Unsupported file type. Please use PDF, DOCX, or XLSX.")

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.
    
    First attempts to extract text directly using pdfplumber. If the extracted
    text length is less than 50 characters, assumes it's a scanned image and
    uses pytesseract (OCR) to extract text instead.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
        
    Raises:
        Exception: If PDF cannot be read or if Tesseract is not installed (OCR fallback).
    """
    try:
        # First, try direct text extraction
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page_num, page in enumerate(pdf.pages, start=1):
                extracted = page.extract_text()
                if extracted:
                    text += f"--- Page {page_num} ---\n{extracted}\n"
        
        # If text extraction yields minimal content, use OCR
        if len(text.strip()) < 50:
            text = _extract_text_from_pdf_ocr(file_path)
        
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF file '{file_path}': {str(e)}")

def _extract_text_from_pdf_ocr(file_path: str) -> str:
    """
    Extracts text from a PDF using OCR (pytesseract).
    
    Converts each PDF page to an image and uses Tesseract to perform
    optical character recognition.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from OCR.
        
    Raises:
        Exception: If pdf2image or pytesseract encounters an error.
    """
    try:
        # Convert PDF pages to images
        images = convert_from_path(file_path)
        ocr_text = ''
        
        for page_num, image in enumerate(images, start=1):
            # Perform OCR on each page image
            page_text = pytesseract.image_to_string(image)
            ocr_text += f"--- Page {page_num} (OCR) ---\n{page_text}\n"
        
        return ocr_text
    except Exception as e:
        raise Exception(f"Error during OCR extraction: {str(e)}")

def extract_text_from_docx(file_path: str) -> str:
    """
    Extracts text from a DOCX file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text.
        
    Raises:
        Exception: If DOCX file cannot be read or parsed.
    """
    try:
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise Exception(f"Error reading DOCX file '{file_path}': {str(e)}")

def extract_text_from_xlsx(file_path: str) -> str:
    """
    Extracts text from an XLSX file.
    
    Reads all sheets, removes completely empty rows and columns to optimize
    for LLM processing, and converts to readable string format.

    Args:
        file_path (str): The path to the XLSX file.

    Returns:
        str: The extracted and cleaned data formatted with sheet headers.
        
    Raises:
        Exception: If the XLSX file cannot be read or parsed.
    """
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_contents = []
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Drop completely empty rows and columns
            df = df.dropna(how='all')  # Drop completely empty rows
            df = df.dropna(axis=1, how='all')  # Drop completely empty columns
            
            # Add sheet header
            sheet_contents.append(f"### Sheet: {sheet_name}\n")
            
            # Convert to string
            sheet_text = df.to_string(index=False)
            sheet_contents.append(sheet_text)
            sheet_contents.append("")  # Blank line between sheets
        
        return "\n".join(sheet_contents)
    except Exception as e:
        raise Exception(f"Error reading XLSX file '{file_path}': {str(e)}")

if __name__ == '__main__':
    # Example usage
    test_file = 'sample_invoice.pdf'
    try:
        text = extract_text_from_file(test_file)
        print(text[:500])  # Print first 500 characters
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error: {str(e)}")
