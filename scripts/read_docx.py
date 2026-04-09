import sys
import docx

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        with open('deepfake_report_output.txt', 'w', encoding='utf-8') as f:
            for para in doc.paragraphs:
                f.write(para.text + '\n')
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    extract_text_from_docx('deepfake_final_report_981dim.docx')
