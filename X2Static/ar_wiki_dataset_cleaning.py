# ar_wiki_dataset_cleaning.py
import os
import json
import re
from camel_tools.utils.charmap import CharMapper

# Directory and output file settings
EXTRACTED_DIR = './extracted_arwiki'                # Folder containing extracted Wikipedia dumps (from WikiExtractor)
SENTENCE_CORPUS_FILE = 'corpus_arabic_sentences.txt'  # Output file: one sentence per line
PARAGRAPH_CORPUS_FILE = 'corpus_arabic_paragraphs.txt'  # Output file: one paragraph per line

# Initialize the CAMeL Tools "arclean" character normalizer (recommended by aubmindlab for Arabic)
normalizer = CharMapper.builtin_mapper('arclean')

def remove_unwanted(text):
    """
    Remove unwanted content from text:
      - URLs (http...)
      - Reference markers like [1], [12], etc.
      - All digits (Arabic/Western)
    
    Args:
        text (str): Input text.
    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'http\S+', '', text)                 # Remove URLs
    text = re.sub(r'\[\s*\d+\s*\]', '', text)           # Remove reference markers
    text = re.sub(r'\d+', '', text)                     # Remove digits
    return text

def normalize_text(text):
    """
    Normalize and clean Arabic text:
      - Apply 'arclean' mapping (standardizes common Arabic spelling/diacritics)
      - Remove elongation character (ـ)
      - Remove unwanted items (URLs, references, digits)
      - Collapse multiple spaces
      - Retain only Arabic letters, whitespace, and allowed punctuation: . ؟ !
    
    Args:
        text (str): Raw input text.
    Returns:
        str: Normalized, filtered text.
    """
    norm = normalizer.map_string(text)
    norm = norm.replace("ـ", "")  # Remove elongation marks (kashida)
    norm = remove_unwanted(norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    # Allow only Arabic Unicode block, whitespace, and specified punctuation
    allowed_punct = r'\.\؟!'
    norm = re.sub(fr'[^\u0600-\u06FF\s{allowed_punct}]', '', norm)
    return norm

def segment_sentences(text):
    """
    Split normalized text into sentences based on punctuation (. ؟ !).

    Args:
        text (str): Normalized text string.
    Returns:
        List[str]: List of sentences.
    """
    sentences = re.split(r'(?<=[\.؟!])\s+', text)
    # Remove empty results and trim whitespace
    return [s.strip() for s in sentences if s.strip()]

def segment_paragraphs(text):
    """
    Split normalized text into paragraphs, using blank lines as separators.

    Args:
        text (str): Normalized text string.
    Returns:
        List[str]: List of paragraphs.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def process_file(filepath):
    """
    Process a single file from the extracted Wikipedia dump.
    Handles both WikiExtractor --json lines and raw text files.

    Args:
        filepath (str): Path to file to process.
    Returns:
        List[str]: List of article texts.
    """
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Try to parse JSON lines (if --json flag was used with WikiExtractor)
                article = json.loads(line)
                text = article.get('text', '')
                if text:
                    articles.append(text)
            except json.JSONDecodeError:
                # If not JSON, treat as raw text
                articles.append(line)
    return articles

def create_corpora():
    """
    Main function:
    Iterates through all files in the extracted_arwiki directory,
    cleans and normalizes each article, splits into sentences and paragraphs,
    and writes results to the respective output files.

    Outputs:
        - One sentence per line in SENTENCE_CORPUS_FILE
        - One paragraph per line in PARAGRAPH_CORPUS_FILE
    """
    # Open output files (overwrite if exist)
    with open(SENTENCE_CORPUS_FILE, 'w', encoding='utf-8') as out_sent, \
         open(PARAGRAPH_CORPUS_FILE, 'w', encoding='utf-8') as out_para:
        
        file_count = 0
        # Traverse extracted_arwiki directory recursively
        for root, dirs, files in os.walk(EXTRACTED_DIR):
            for filename in files:
                file_count += 1
                filepath = os.path.join(root, filename)
                articles = process_file(filepath)
                for article in articles:
                    if not article.strip():
                        continue
                    # Normalize and clean article text
                    norm_text = normalize_text(article)
                    # Segment and write sentences
                    sentences = segment_sentences(norm_text)
                    for s in sentences:
                        out_sent.write(s + "\n")
                    # Segment and write paragraphs
                    paragraphs = segment_paragraphs(norm_text)
                    for p in paragraphs:
                        out_para.write(p + "\n")
                # Progress report every 100 files
                if file_count % 100 == 0:
                    out_sent.flush()
                    out_para.flush()
                    print(f"Processed {file_count} files...")
        
        print(f"Finished processing {file_count} files.")
    
    print(f"Sentence corpus written to {SENTENCE_CORPUS_FILE}")
    print(f"Paragraph corpus written to {PARAGRAPH_CORPUS_FILE}")

if __name__ == '__main__':
    create_corpora()
