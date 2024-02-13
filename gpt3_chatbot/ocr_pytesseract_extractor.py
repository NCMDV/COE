import os
import pytesseract as tess
from PIL import Image
from pdf2image import convert_from_path

tess.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")

# This function takes a PDF file name as input and returns the name of the text file that contains the extracted text.
def read_pdf(file_path):
    # Store all pages of one file here
    pages = []

    try:
        # Convert the PDF file to a list of PIL images:
        images = convert_from_path(file_path)

        # Extract text from each image:
        for i, image in enumerate(images):
            # Generating filename for each image
            filename = "page_" + str(i) + "_" + os.path.basename(file_path) + ".jpeg"
            # Save the image with page number
            image.save(filename, "JPEG")

            text = tess.image_to_string(Image.open(filename)) # Extracting text from each image using pytesseract
            pages.append(text)
            
    except Exception as e:
        print(str(e))

    # Write the extracted text to a file:
    output_file_name = os.path.splitext(file_path)[0] + ".txt" # Generating output file name

    with open(output_file_name, "w") as f:
        # Writing extracted text to output file
        f.write("\n".join(pages))

    return output_file_name


pdf_file = "./data/docs/HR Downloadable Forms.pdf"
print(read_pdf(pdf_file))