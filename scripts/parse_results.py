import sys
import glob
from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_pre = False
        self.text_blocks = []
        self.current_block = []

    def handle_starttag(self, tag, attrs):
        if tag == 'pre':
            self.in_pre = True

    def handle_endtag(self, tag):
        if tag == 'pre':
            self.in_pre = False
            if self.current_block:
                text = "".join(self.current_block).strip()
                if len(text) > 0 and 'import ' not in text and 'def ' not in text:
                    self.text_blocks.append(text)
                self.current_block = []

    def handle_data(self, data):
        if self.in_pre:
            self.current_block.append(data)

def process(filepath, out_file):
    out_file.write(f"\n======================================\n")
    out_file.write(f"  {filepath}\n")
    out_file.write(f"======================================\n")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    parser = TextExtractor()
    parser.feed(content)
    
    # Also grab table text if we missed it
    
    for block in parser.text_blocks:
        out_file.write(block + '\n')
        out_file.write("-" * 40 + '\n')

if __name__ == '__main__':
    with open('parsed_html_results.txt', 'w', encoding='utf-8') as out_f:
        for f in sorted(glob.glob('*.html')):
            process(f, out_f)
