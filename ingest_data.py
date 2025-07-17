import io
import base64
from typing import List
from pathlib import Path
from PIL import Image
import pytesseract
import fitz  
import logging
import google.generativeai as genai
import os
from dataclasses import dataclass

from llama_index.core.schema import Document
from llama_parse import LlamaParse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@dataclass
class ImageAnalysis:
    description: str
    extracted_text: str
    data_tables: List[str]
    charts_info: List[str]
    confidence_score: float

class MultimodalDocumentProcessor:

    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        genai.configure(api_key=google_api_key)
        # Use the latest supported Gemini model for vision
        self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
        
    def analyze_image_with_vision(self, image: Image.Image, context: str = "") -> ImageAnalysis:

        try:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create detailed prompt for financial/economic document analysis
            prompt = f"""
            Analyze this image from a financial/economic document. Please provide:
            
            1. DETAILED DESCRIPTION: What type of visualization is this? (chart, graph, table, etc.)
            2. EXTRACTED TEXT: All visible text including titles, labels, legends, and data values
            3. DATA EXTRACTION: If this contains numerical data, extract it in structured format
            4. CHART/GRAPH ANALYSIS: If this is a chart/graph, describe:
               - What metric is being measured
               - Time periods covered
               - Key trends and patterns
               - Specific data points and values
            5. TABLE DATA: If this is a table, extract the complete data structure
            
            Context: {context}
            
            Focus especially on:
            - GDP growth rates and economic indicators
            - Historical financial data
            - Time series information
            - Percentages and numerical values
            
            Format your response as:
            DESCRIPTION: [description]
            TEXT: [extracted text]
            DATA: [structured data]
            ANALYSIS: [detailed analysis]
            """
            
            # Analyze with Gemini Vision (latest model)
            response = self.vision_model.generate_content([prompt, image])
            
            if response and response.text:
                analysis_text = response.text
                
                # Parse the structured response
                description = self._extract_section(analysis_text, "DESCRIPTION:")
                extracted_text = self._extract_section(analysis_text, "TEXT:")
                data_section = self._extract_section(analysis_text, "DATA:")
                analysis_section = self._extract_section(analysis_text, "ANALYSIS:")
                
                # Combine all extracted information
                full_text = f"{description}\n\n{extracted_text}\n\n{data_section}\n\n{analysis_section}"
                
                return ImageAnalysis(
                    description=description,
                    extracted_text=full_text,
                    data_tables=[data_section] if data_section else [],
                    charts_info=[analysis_section] if analysis_section else [],
                    confidence_score=0.9
                )
            
        except Exception as e:
            logger.error(f"Vision API analysis failed: {str(e)}")
            
        # Fallback to OCR if vision fails
        return self._fallback_ocr_analysis(image)
    
    def _extract_section(self, text: str, section_header: str) -> str:

        lines = text.split('\n')
        section_lines = []
        in_section = False
        
        for line in lines:
            if line.strip().startswith(section_header):
                in_section = True
                section_lines.append(line.replace(section_header, '').strip())
            elif in_section and line.strip().startswith(('DESCRIPTION:', 'TEXT:', 'DATA:', 'ANALYSIS:')):
                break
            elif in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines).strip()
    
    def _fallback_ocr_analysis(self, image: Image.Image) -> ImageAnalysis:

        try:
            # OCR for financial documents
            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(image, lang='eng', config=custom_config)
            
            return ImageAnalysis(
                description="OCR-extracted content",
                extracted_text=extracted_text,
                data_tables=[],
                charts_info=[],
                confidence_score=0.5
            )
        except Exception as e:
            logger.error(f"OCR fallback failed: {str(e)}")
            return ImageAnalysis("", "", [], [], 0.0)

def load_and_parse_multimodal_documents(data_dir: str, parser: LlamaParse, google_api_key: str) -> List[Document]:

    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory '{data_dir}' does not exist")
        return documents

    # Initialize multimodal processor
    processor = MultimodalDocumentProcessor(google_api_key)

    # Supported file extensions
    pdf_extensions = {'.pdf'}
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}

    # Get all files
    all_files = [f for f in data_path.rglob('*') if f.is_file()]

    if not all_files:
        logger.warning(f"No files found in '{data_dir}'")
        return documents

    logger.info(f"Found {len(all_files)} files in '{data_dir}'")

    pdf_files = [f for f in all_files if f.suffix.lower() in pdf_extensions]
    if pdf_files:
        logger.info(f"Processing {len(pdf_files)} PDF files with multimodal analysis...")
        documents.extend(_process_pdf_files_multimodal(pdf_files, parser, processor))

    # Process standalone image files
    image_files = [f for f in all_files if f.suffix.lower() in image_extensions]
    if image_files:
        logger.info(f"Processing {len(image_files)} image files...")
        documents.extend(_process_image_files_multimodal(image_files, processor))

    logger.info(f"Total documents processed: {len(documents)}")
    return documents

def _process_pdf_files_multimodal(pdf_files: List[Path], parser: LlamaParse, processor: MultimodalDocumentProcessor) -> List[Document]:

    documents = []
    
    for pdf_file in pdf_files:
        logger.info(f"Processing PDF with multimodal analysis: {pdf_file.name}")
        
        # Extract text content first
        text_docs = _extract_text_from_pdf(pdf_file, parser)
        
        # Extract and analyze images from PDF
        image_docs = _extract_and_analyze_pdf_images(pdf_file, processor)
        
        # Combine text and image analysis
        combined_content = []
        
        # Add text content
        for doc in text_docs:
            if doc.text and _is_meaningful_text(doc.text):
                combined_content.append(f"TEXT CONTENT:\n{doc.text}")
        
        # Add image analysis
        for doc in image_docs:
            if doc.text and _is_meaningful_text(doc.text):
                combined_content.append(f"IMAGE ANALYSIS:\n{doc.text}")
        
        # Create combined document
        if combined_content:
            combined_text = "\n\n" + "="*50 + "\n\n".join(combined_content)
            
            combined_doc = Document(
                text=combined_text,
                metadata={
                    "file_name": pdf_file.name,
                    "file_path": str(pdf_file),
                    "file_type": "pdf_multimodal",
                    "file_size": pdf_file.stat().st_size,
                    "parsing_method": "multimodal_analysis",
                    "text_sections": len(text_docs),
                    "image_sections": len(image_docs)
                }
            )
            documents.append(combined_doc)
    
    return documents

def _extract_text_from_pdf(pdf_file: Path, parser: LlamaParse) -> List[Document]:

    try:
        logger.info(f"Extracting text with LlamaParse: {pdf_file.name}")
        # Use system_prompt instead of deprecated parsing_instruction
        docs = parser.load_data(str(pdf_file), system_prompt=(
            "Extract all content from this document including text, tables, charts, and graphs. "
            "Pay special attention to numerical data, financial metrics, and economic indicators. "
            "Preserve the structure and context of tables and charts. "
            "If you encounter charts or graphs, describe them in detail including axis labels, data points, and trends."
        ))
        if docs and _is_meaningful_text(docs[0].text):
            return docs
    except Exception as e:
        logger.warning(f"LlamaParse failed for {pdf_file.name}: {str(e)}")

    # Fallback to PyMuPDF
    try:
        logger.info(f"Extracting text with PyMuPDF: {pdf_file.name}")
        return _parse_with_pymupdf(pdf_file)
    except Exception as e:
        logger.warning(f"PyMuPDF failed for {pdf_file.name}: {str(e)}")

    return []

def _extract_and_analyze_pdf_images(pdf_file: Path, processor: MultimodalDocumentProcessor) -> List[Document]:

    documents = []
    
    try:
        pdf_document = fitz.open(str(pdf_file))
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract images from the page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Analyze with vision model
                        context = f"Page {page_num + 1} of {pdf_file.name}"
                        analysis = processor.analyze_image_with_vision(image, context)
                        
                        if analysis.extracted_text and _is_meaningful_text(analysis.extracted_text):
                            doc = Document(
                                text=analysis.extracted_text,
                                metadata={
                                    "file_name": pdf_file.name,
                                    "file_path": str(pdf_file),
                                    "file_type": "pdf_image_analysis",
                                    "page_number": page_num + 1,
                                    "image_index": img_index,
                                    "parsing_method": "vision_analysis",
                                    "confidence_score": analysis.confidence_score,
                                    "image_description": analysis.description
                                }
                            )
                            documents.append(doc)
                            logger.info(f"Analyzed image {img_index} from page {page_num + 1} of {pdf_file.name}")
                    
                    pix = None  # Clean up
                    
                except Exception as e:
                    logger.warning(f"Error processing image {img_index} from page {page_num + 1}: {str(e)}")
                    continue
        
        pdf_document.close()
        
    except Exception as e:
        logger.error(f"Error extracting images from {pdf_file.name}: {str(e)}")
    
    return documents

def _process_image_files_multimodal(image_files: List[Path], processor: MultimodalDocumentProcessor) -> List[Document]:

    documents = []
    
    for image_file in image_files:
        try:
            logger.info(f"Processing image with multimodal analysis: {image_file.name}")
            
            with Image.open(image_file) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Analyze with vision model
                analysis = processor.analyze_image_with_vision(img, f"Standalone image: {image_file.name}")
                
                if analysis.extracted_text and _is_meaningful_text(analysis.extracted_text):
                    doc = Document(
                        text=analysis.extracted_text,
                        metadata={
                            "file_name": image_file.name,
                            "file_path": str(image_file),
                            "file_type": "image_multimodal",
                            "file_size": image_file.stat().st_size,
                            "image_dimensions": f"{img.width}x{img.height}",
                            "parsing_method": "vision_analysis",
                            "confidence_score": analysis.confidence_score,
                            "image_description": analysis.description
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Successfully analyzed {image_file.name}")
                
        except Exception as e:
            logger.error(f"Error processing {image_file.name}: {str(e)}")
            continue
    
    return documents

def _parse_with_pymupdf(pdf_file: Path) -> List[Document]:

    documents = []
    
    try:
        pdf_document = fitz.open(str(pdf_file))
        full_text = ""
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            if text.strip():
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        pdf_document.close()
        
        if full_text.strip():
            doc = Document(
                text=full_text,
                metadata={
                    "file_name": pdf_file.name,
                    "file_path": str(pdf_file),
                    "file_type": "pdf_text",
                    "file_size": pdf_file.stat().st_size,
                    "parsing_method": "pymupdf_text"
                }
            )
            documents.append(doc)
    
    except Exception as e:
        logger.error(f"PyMuPDF parsing failed for {pdf_file.name}: {str(e)}")
    
    return documents

def _is_meaningful_text(text: str) -> bool:

    if not text or len(text.strip()) < 20:
        return False
    
    # Count letters vs other characters
    letter_count = sum(1 for c in text if c.isalpha())
    total_count = len(text)
    
    # Text should have at least 20% letters
    return letter_count / total_count > 0.2

def validate_multimodal_setup():

    try:
        # Check Tesseract
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR version: {version}")

        # Check Google API key
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables")
            return False

        # Test vision model with latest supported model
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Google Gemini Vision API (gemini-2.5-flash) configured successfully")

        return True

    except Exception as e:
        logger.error(f"Multimodal setup validation failed: {str(e)}")
        return False