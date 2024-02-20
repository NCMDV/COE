# Automated Live Chatbot Intelligent Agent (ALiCIA)

An Intelligent Chatbot built by 77Global.

--------------------
### Steps to make `pytesseract` work for Image OCR
- Install poppler for Windows
  - Download the latest poppler version @ [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
  - Extract to desired place on your system (C:/ProgramFiles)
  - Add the `bin/` directory to System PATH
- Install tessearct for Windows
  - Download and install tesseract for windows @ [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
  - `pip install pytesseract`
  - Add tesseract path as environment variable `TESSERACT_PATH`
  - Set the tesseract path in the script
    `pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")`
