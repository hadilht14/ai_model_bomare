# # pdf_utils.py
# import os
# import logging
# import pytesseract
# from pdf2image import convert_from_path, pdfinfo_from_path
# from PIL import Image # Pillow is a dependency
# import pypdf # Using pypdf for text-based PDFs due to its performance and compatibility with encrypted and complex PDFs
# import asyncio # For asyncio.to_thread

# log = logging.getLogger(__name__)

# # --- Tesseract Configuration ---
# # (Keep your existing Tesseract and Poppler configuration comments and attempts here)
# try:
#     tesseract_version = pytesseract.get_tesseract_version()
#     log.info(f"Tesseract OCR version {tesseract_version} found and configured.")
# except pytesseract.TesseractNotFoundError:
#     log.critical(
#         "Tesseract OCR is not installed or not found in your PATH. "
#         "PDF OCR functionality (for image-based PDFs) will not work. "
#         "Please install Tesseract and add it to your PATH, or set the TESSERACT_CMD environment variable."
#     )
#     # To allow the program to run without Tesseract for text-based PDFs:
#     tesseract_version = None # Indicate Tesseract is not available
# except Exception as e:
#     log.error(f"An unexpected error occurred while checking Tesseract version: {e}")
#     tesseract_version = None


# # --- Poppler Configuration ---
# # (Keep your existing Poppler configuration comments here)
# # Example for manual Poppler path (uncomment and adjust if needed):
# # POPPLER_PATH_WINDOWS = r"C:\path\to\poppler-version\bin" # Change to your Poppler /bin directory
# # if os.name == 'nt' and not os.getenv("POPPLER_PATH") and POPPLER_PATH_WINDOWS and os.path.isdir(POPPLER_PATH_WINDOWS) :
# #      # Note: This only sets it for the current process if Poppler is not in system PATH.
# #      # pdf2image might still need poppler_path explicitly if this doesn't work globally.
# #      os.environ["PATH"] += os.pathsep + POPPLER_PATH_WINDOWS
# #      log.info(f"Attempted to add Poppler path to system PATH: {POPPLER_PATH_WINDOWS}")


# async def _extract_text_pypdf(pdf_path: str) -> str:
#     text_from_pypdf = ""
#     try:
#         reader = pypdf.PdfReader(pdf_path)
#         if reader.is_encrypted:
#             try:
#                 # Try empty password for owner or user
#                 decrypted_owner = await asyncio.to_thread(reader.decrypt, "") == pypdf.PasswordType.OWNER_PASSWORD
#                 if decrypted_owner:
#                     log.info(f"PDF '{pdf_path}' successfully decrypted with empty owner password.")
#                 else:
#                     # Need to re-check reader.is_encrypted as decrypt changes state
#                     if reader.is_encrypted: # Check again if still encrypted
#                         decrypted_user = await asyncio.to_thread(reader.decrypt, "") == pypdf.PasswordType.USER_PASSWORD
#                         if decrypted_user:
#                             log.info(f"PDF '{pdf_path}' successfully decrypted with empty user password.")
#                         elif reader.is_encrypted: # Still encrypted after trying empty user password
#                              log.warning(f"PDF '{pdf_path}' is encrypted and decryption failed. OCR might also fail or yield garbage.")
#             except Exception as decrypt_e:
#                 log.warning(f"Failed to decrypt PDF '{pdf_path}': {decrypt_e}. Text extraction might be incomplete.")
        
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             extracted = await asyncio.to_thread(page.extract_text) # Run blocking call in thread
#             if extracted:
#                 text_from_pypdf += extracted + "\n"
#         return text_from_pypdf.strip()
#     except pypdf.errors.PdfReadError as e:
#         log.warning(f"pypdf could not read '{pdf_path}' (possibly corrupted or complex PDF): {e}. Will rely on OCR if available.")
#         return ""
#     except Exception as e:
#         log.error(f"Error using pypdf for '{pdf_path}': {e}. Will rely on OCR if available.")
#         return ""

# async def _ocr_pdf_pages(pdf_path: str, poppler_path_cfg: str | None = None) -> str:
#     full_ocr_text_parts = []
#     try:
#         # Use poppler_path if configured, otherwise let pdf2image find it.
#         # This is where you'd pass your POPPLER_PATH_WINDOWS if needed and available
#         # images = await asyncio.to_thread(convert_from_path, pdf_path, poppler_path=POPPLER_PATH_WINDOWS or None)
#         images = await asyncio.to_thread(convert_from_path, pdf_path, poppler_path=poppler_path_cfg)

#         if not images:
#             log.warning(f"pdf2image returned no images for OCR from PDF: {pdf_path}")
#             return ""

#         for i, image in enumerate(images):
#             log.debug(f"OCR'ing page {i+1} of {len(images)} from '{pdf_path}'...")
#             try:
#                 page_text = await asyncio.to_thread(pytesseract.image_to_string, image) # OCR in thread
#                 if page_text:
#                     full_ocr_text_parts.append(page_text.strip())
#             except pytesseract.TesseractError as e_tess_ocr:
#                 log.error(f"Tesseract OCR error on page {i+1} of '{pdf_path}': {e_tess_ocr}")
#             finally:
#                 image.close()
        
#         return "\n\n".join(filter(None, full_ocr_text_parts)).strip()
#     except Exception as e_conv: # Catches pdf2image errors e.g. Poppler missing
#         err_conv = f"Failed to convert PDF '{pdf_path}' to images for OCR: {e_conv}"
#         log.error(err_conv, exc_info=True)
#         if "poppler" in str(e_conv).lower() or "pdfinfo not installed" in str(e_conv).lower():
#             log.error("OCR Error: PDF processing utilities (like Poppler) seem to be missing or not configured.")
#         return ""


# async def extract_text_from_pdf(pdf_path: str) -> tuple[str | None, str | None]:
#     """
#     Extracts text from a PDF file.
#     Tries pypdf for text-based extraction.
#     Also tries OCR if Tesseract is available.
#     Returns the better result or an error.

#     Returns:
#         tuple[str | None, str | None]: (extracted_text, error_message)
#         extracted_text is None if a critical error occurs or no text is found.
#         error_message contains details if an error occurred.
#     """
#     if not os.path.exists(pdf_path):
#         err = f"PDF file not found at path: {pdf_path}"
#         log.error(err)
#         return None, err
    
#     log.info(f"Attempting to extract text from PDF: {pdf_path}")
    
#     text_from_pypdf = ""
#     text_from_ocr = ""
#     pypdf_error = None
#     ocr_error = None

#     # 1. Try pypdf (direct text extraction)
#     try:
#         text_from_pypdf = await _extract_text_pypdf(pdf_path)
#         if text_from_pypdf:
#             log.info(f"pypdf extracted {len(text_from_pypdf)} chars from '{pdf_path}'.")
#         else:
#             log.info(f"pypdf extracted no text from '{pdf_path}'.")
#     except Exception as e: # Should be caught by _extract_text_pypdf, but as safeguard
#         pypdf_error = f"pypdf extraction failed: {e}"
#         log.error(pypdf_error)

#     # 2. Try OCR (if Tesseract is available)
#     if tesseract_version: # Check if Tesseract was found during init
#         log.info(f"Attempting OCR for PDF: {pdf_path}")
#         # POPPLER_PATH_CONFIG = os.getenv("POPPLER_PATH_BIN") # Or your specific config
#         POPPLER_PATH_CONFIG = None # Set this if you have a specific Poppler bin path
#         # Example: POPPLER_PATH_CONFIG = r"C:\path\to\poppler-version\bin"
#         try:
#             text_from_ocr = await _ocr_pdf_pages(pdf_path, POPPLER_PATH_CONFIG)
#             if text_from_ocr:
#                 log.info(f"OCR extracted {len(text_from_ocr)} chars from '{pdf_path}'.")
#             else:
#                 log.info(f"OCR extracted no text from '{pdf_path}'.")
#         except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as e_tess:
#             ocr_error = f"Tesseract OCR runtime error for '{pdf_path}': {e_tess}"
#             log.critical(ocr_error)
#         except Exception as e_ocr_generic:
#             ocr_error = f"Generic OCR processing error for '{pdf_path}': {e_ocr_generic}"
#             log.error(ocr_error, exc_info=True)
#     else:
#         log.info("Tesseract OCR not available or not configured. Skipping OCR.")
#         ocr_error = "Tesseract OCR not available." # Not a fatal error for the function, just info

#     # 3. Decide which text to return
#     if text_from_ocr and len(text_from_ocr) > len(text_from_pypdf) + 100: # Prefer OCR if substantially more text
#         log.info(f"Using OCR result (len {len(text_from_ocr)}) as it's substantially more than pypdf (len {len(text_from_pypdf)}).")
#         return text_from_ocr, None
#     elif text_from_pypdf:
#         log.info(f"Using pypdf result (len {len(text_from_pypdf)}). OCR result was (len {len(text_from_ocr)}).")
#         return text_from_pypdf, (ocr_error if text_from_ocr and not text_from_pypdf else None) # Return ocr_error as warning if pypdf worked but ocr failed
#     elif text_from_ocr: # pypdf got nothing, but OCR did
#         log.info(f"Using OCR result (len {len(text_from_ocr)}) as pypdf found no text.")
#         return text_from_ocr, pypdf_error # pypdf_error might be "pypdf extracted no text" or an actual error
#     else: # Neither got anything
#         final_error_msg = "No text could be extracted from the PDF using available methods."
#         if pypdf_error and ocr_error:
#             final_error_msg += f" pypdf: {pypdf_error}. OCR: {ocr_error}."
#         elif pypdf_error:
#             final_error_msg += f" pypdf: {pypdf_error}."
#         elif ocr_error:
#             final_error_msg += f" OCR: {ocr_error}."
#         log.warning(f"Neither pypdf nor OCR extracted text from '{pdf_path}'. Errors: pypdf='{pypdf_error}', ocr='{ocr_error}'")
#         return None, final_error_msg