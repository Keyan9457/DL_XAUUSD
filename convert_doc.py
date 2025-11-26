import markdown
from xhtml2pdf import pisa

def convert_md_to_pdf(source_md, output_pdf):
    # 1. Read Markdown
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 2. Convert to HTML
    html_content = markdown.markdown(text)
    
    # 3. Add Styling
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Helvetica, sans-serif; font-size: 12pt; line-height: 1.5; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 20px; border-bottom: 1px solid #eee; }}
            h3 {{ color: #7f8c8d; margin-top: 15px; }}
            ul {{ margin-bottom: 10px; }}
            li {{ margin-bottom: 5px; }}
            code {{ background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: Courier; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # 4. Convert to PDF
    with open(output_pdf, "wb") as result_file:
        pisa_status = pisa.CreatePDF(styled_html, dest=result_file)
        
    if pisa_status.err:
        print(f"Error converting to PDF: {pisa_status.err}")
    else:
        print(f"Successfully created {output_pdf}")

if __name__ == "__main__":
    convert_md_to_pdf(
        r"C:\Users\Admin\.gemini\antigravity\brain\95197eef-e76d-4aba-ab3e-c34440d2762c\project_documentation.md",
        r"C:\Users\Admin\.gemini\antigravity\brain\95197eef-e76d-4aba-ab3e-c34440d2762c\project_documentation.pdf"
    )
