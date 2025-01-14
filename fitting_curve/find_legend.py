import fitz  # PyMuPDF

def extract_bottom_left_vector(pdf_path, output_path):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    output_pdf = fitz.open()  # Create a new PDF

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        # Get the page dimensions
        rect = page.rect
        # Define the bottom-left 15% region
        new_width = rect.width * 0.12
        new_height = rect.height * 0.06
        bottom_left_rect = fitz.Rect(0, rect.height - new_height, new_width, rect.height)

        # Create a new page with the cropped area in the output PDF
        new_page = output_pdf.new_page(width=new_width, height=new_height)
        
        # Extract vector content by copying the clip region
        new_page.show_pdf_page(fitz.Rect(0, 0, new_width, new_height), pdf_document, page_number, clip=bottom_left_rect)

    # Save the new PDF
    output_pdf.save(output_path)
    pdf_document.close()
    output_pdf.close()
    print(f"Bottom-left 15% saved to {output_path}")

# Example usage
extract_bottom_left_vector("ori.pdf", "output.pdf")
