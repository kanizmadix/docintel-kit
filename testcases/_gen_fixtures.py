"""One-off script to generate and sanity-check fixtures without pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import conftest

d = conftest._ensure_dir()
conftest._make_text_pdf(d / "text_report.pdf")
conftest._make_empty_pdf(d / "empty_page.pdf")
conftest._make_image_only_pdf(d / "scanned_invoice.pdf")
conftest._make_table_pdf(d / "invoice_table.pdf")
conftest._make_corrupted_pdf(d / "corrupted.pdf")
conftest._make_truncated_pdf(d / "text_report.pdf", d / "truncated.pdf")
conftest._make_docx(d / "contract.docx")
conftest._make_pptx(d / "slides.pptx")
conftest._make_malformed_html(d / "malformed.html")
form_image = d / "form.png"
conftest._make_form_image(form_image)
conftest._make_rotated_image(form_image, d / "form_rotated.png")
conftest._make_blank_image(d / "blank.png")
conftest._make_corrupted_image(d / "corrupted.png")
conftest._make_xlsx(d / "workbook.xlsx")
conftest._make_csv(d / "tricky.csv")
conftest._make_tsv(d / "tricky.tsv")
conftest._make_unsupported_file(d / "unknown.xyz")

for f in sorted(d.iterdir()):
    print(f.name, f.stat().st_size, "bytes")
