import fitz # PyMuPDF
import sys

# --- 1. SET YOUR PDF FILENAME HERE ---
# (Place a copy of your sample PDF in the same folder)
PDF_FILENAME = "YOUR_SAMPLE_PDF.pdf" 

try:
    pdf_document = fitz.open(PDF_FILENAME)
except Exception:
    print(f"Error: Could not open file '{PDF_FILENAME}'.")
    print("Please make sure the file is in the same folder as this script.")
    sys.exit(1)

print(f"--- F-I-E-L-D---R-E-P-O-R-T for {PDF_FILENAME} ---")
print("Found the following fillable form fields:\n")

field_count = 0
for page_num, page in enumerate(pdf_document):
    for widget in page.widgets():
        field_count += 1
        print(f"  Field Name: '{widget.field_name}'")
        print(f"  Field Type: {widget.field_type_string}")
        print(f"  Current Value: '{widget.field_value}'")
        print("  ---")

if field_count == 0:
    print("This PDF has NO fillable form fields (AcroForm).")
    print("You will need to rely 100% on the Regex method (Attempt 2).")
else:
    print(f"\nFound {field_count} total fields.")
    print("\nUse the 'Field Name' values in your FIELD_NAME_MAP in main.py.")

pdf_document.close()