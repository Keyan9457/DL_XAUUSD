from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Project Summary: AI-Powered XAUUSD Trading Bot', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def create_pdf(input_file, output_file):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(2)
            continue

        if line.startswith('# '):
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, line[2:], 0, 1, 'L')
            pdf.ln(2)
        elif line.startswith('## '):
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, line[3:], 0, 1, 'L')
            pdf.ln(1)
        elif line.startswith('### '):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, line[4:], 0, 1, 'L')
            pdf.ln(1)
        elif line.startswith('![') and line.endswith(')'):
            # Image handling: ![AltText](filename.png)
            try:
                start_idx = line.find('(') + 1
                end_idx = line.find(')')
                image_path = line[start_idx:end_idx]
                
                if os.path.exists(image_path):
                    # Get page width
                    page_width = pdf.w - 2 * pdf.l_margin
                    # Add image centered and scaled to fit width
                    pdf.image(image_path, x=pdf.l_margin, w=page_width)
                    pdf.ln(5)
                else:
                    print(f"Warning: Image not found: {image_path}")
            except Exception as e:
                print(f"Error adding image: {e}")
        elif line.startswith('* ') or line.startswith('- '):
            pdf.set_font('Arial', '', 11)
            pdf.cell(10) # Indent
            pdf.multi_cell(0, 6, chr(149) + ' ' + line[2:])
        elif line.startswith('    * ') or line.startswith('    - '):
             pdf.set_font('Arial', '', 11)
             pdf.cell(20) # Double Indent
             pdf.multi_cell(0, 6, chr(149) + ' ' + line[6:])
        elif line.startswith('1. ') or (len(line) > 3 and line[1] == '.' and line[2] == ' '):
             pdf.set_font('Arial', '', 11)
             pdf.multi_cell(0, 6, line)
        elif line.startswith('```'):
            continue # Skip code block markers for now
        else:
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 6, line)

    pdf.output(output_file)
    print(f"PDF generated: {output_file}")

if __name__ == '__main__':
    create_pdf('project_summary.md', 'project_summary.pdf')
