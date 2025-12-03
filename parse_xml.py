#!/usr/bin/env python3
"""
Parse medRxiv JATS XML files and extract text content.

Usage:
    python3 parse_xml.py xml/some-file.xml           # Parse single file
    python3 parse_xml.py --doi 10.1101/2024.02.20.24303046  # Find by DOI
    python3 parse_xml.py --build-index               # Build DOI->file index
"""

import xml.etree.ElementTree as ET
import json
import argparse
import re
import os
import html
from pathlib import Path
from typing import Optional, Dict, List


# HTML entities that appear in medRxiv XMLs but aren't defined in XML
HTML_ENTITIES = {
    '&ndash;': '–',
    '&mdash;': '—',
    '&rsquor;': ''',
    '&lsquor;': ''',
    '&ldquo;': '"',
    '&rdquo;': '"',
    '&prime;': '′',
    '&Prime;': '″',
    '&reg;': '®',
    '&trade;': '™',
    '&copy;': '©',
    '&deg;': '°',
    '&plusmn;': '±',
    '&times;': '×',
    '&divide;': '÷',
    '&sol;': '/',
    '&apos;': "'",
    # Accented characters
    '&eacute;': 'é', '&egrave;': 'è', '&euml;': 'ë', '&ecirc;': 'ê',
    '&aacute;': 'á', '&agrave;': 'à', '&auml;': 'ä', '&acirc;': 'â', '&atilde;': 'ã',
    '&iacute;': 'í', '&igrave;': 'ì', '&iuml;': 'ï', '&icirc;': 'î',
    '&oacute;': 'ó', '&ograve;': 'ò', '&ouml;': 'ö', '&ocirc;': 'ô', '&otilde;': 'õ',
    '&uacute;': 'ú', '&ugrave;': 'ù', '&uuml;': 'ü', '&ucirc;': 'û',
    '&ccedil;': 'ç', '&ntilde;': 'ñ', '&szlig;': 'ß',
}


def fix_html_entities(xml_content: str) -> str:
    """Replace HTML entities with their Unicode equivalents."""
    for entity, char in HTML_ENTITIES.items():
        xml_content = xml_content.replace(entity, char)
    return xml_content

XML_DIR = Path("xml")
INDEX_FILE = Path("doi_to_xml.json")


def extract_text_from_element(elem, include_tail=True) -> str:
    """Recursively extract text from an XML element."""
    text_parts = []
    if elem.text:
        text_parts.append(elem.text)
    for child in elem:
        text_parts.append(extract_text_from_element(child))
        if include_tail and child.tail:
            text_parts.append(child.tail)
    return " ".join(text_parts)


def parse_jats_xml(xml_path: Path) -> Dict:
    """Parse a JATS XML file and extract structured content."""
    # Read and fix HTML entities before parsing
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = fix_html_entities(content)
    root = ET.fromstring(content)

    # Handle namespaces
    ns = {
        'xlink': 'http://www.w3.org/1999/xlink',
        'mml': 'http://www.w3.org/1998/Math/MathML',
    }

    result = {
        'file': str(xml_path),
        'doi': None,
        'title': None,
        'authors': [],
        'affiliations': [],
        'abstract': None,
        'body_text': None,
        'categories': [],
        'keywords': [],
    }

    # Find article-meta
    article_meta = root.find('.//article-meta')
    if article_meta is None:
        return result

    # DOI
    doi_elem = article_meta.find('.//article-id[@pub-id-type="doi"]')
    if doi_elem is not None:
        result['doi'] = doi_elem.text

    # Title
    title_elem = article_meta.find('.//title-group/article-title')
    if title_elem is not None:
        result['title'] = extract_text_from_element(title_elem).strip()

    # Authors
    for contrib in article_meta.findall('.//contrib[@contrib-type="author"]'):
        name_elem = contrib.find('.//name')
        if name_elem is not None:
            surname = name_elem.find('surname')
            given = name_elem.find('given-names')
            author = {
                'surname': surname.text if surname is not None else '',
                'given_names': given.text if given is not None else '',
            }
            result['authors'].append(author)

    # Affiliations
    for aff in article_meta.findall('.//aff'):
        aff_text = extract_text_from_element(aff).strip()
        # Remove label numbers
        aff_text = re.sub(r'^\d+\s*', '', aff_text)
        if aff_text:
            result['affiliations'].append(aff_text)

    # Abstract
    abstract_elem = article_meta.find('.//abstract')
    if abstract_elem is not None:
        result['abstract'] = extract_text_from_element(abstract_elem).strip()

    # Categories/subjects
    for subj in article_meta.findall('.//subject'):
        text = subj.text
        if text:
            result['categories'].append(text.strip())

    # Keywords
    for kwd in article_meta.findall('.//kwd'):
        text = kwd.text
        if text:
            result['keywords'].append(text.strip())

    # Body text
    body = root.find('.//body')
    if body is not None:
        sections = []
        for sec in body.findall('.//sec'):
            title_elem = sec.find('title')
            sec_title = title_elem.text if title_elem is not None else ''

            paragraphs = []
            for p in sec.findall('.//p'):
                p_text = extract_text_from_element(p).strip()
                if p_text:
                    paragraphs.append(p_text)

            if paragraphs:
                sections.append({
                    'title': sec_title,
                    'text': '\n\n'.join(paragraphs)
                })

        # If no sections, try getting all paragraphs
        if not sections:
            all_p = []
            for p in body.findall('.//p'):
                p_text = extract_text_from_element(p).strip()
                if p_text:
                    all_p.append(p_text)
            if all_p:
                result['body_text'] = '\n\n'.join(all_p)
        else:
            result['body_text'] = '\n\n'.join(
                f"## {s['title']}\n\n{s['text']}" if s['title'] else s['text']
                for s in sections
            )

    return result


def build_doi_index(xml_dir: Path) -> Dict[str, str]:
    """Build an index mapping DOIs to XML filenames."""
    index = {}
    xml_files = list(xml_dir.glob("*.xml"))
    total = len(xml_files)

    print(f"Building index for {total} files...")

    for i, xml_file in enumerate(xml_files):
        if i % 5000 == 0:
            print(f"  {i}/{total}...")

        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            content = fix_html_entities(content)
            root = ET.fromstring(content)
            doi_elem = root.find('.//article-id[@pub-id-type="doi"]')
            if doi_elem is not None and doi_elem.text:
                index[doi_elem.text] = xml_file.name
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")

    return index


def find_xml_by_doi(doi: str, index: Dict[str, str]) -> Optional[Path]:
    """Find XML file for a given DOI."""
    if doi in index:
        return XML_DIR / index[doi]
    return None


def get_full_text_for_embedding(parsed: Dict) -> str:
    """Combine title, abstract, and body for embedding."""
    parts = []

    if parsed.get('title'):
        parts.append(f"Title: {parsed['title']}")

    if parsed.get('authors'):
        author_names = [f"{a['given_names']} {a['surname']}" for a in parsed['authors']]
        parts.append(f"Authors: {', '.join(author_names)}")

    if parsed.get('categories'):
        parts.append(f"Category: {', '.join(parsed['categories'])}")

    if parsed.get('abstract'):
        parts.append(f"Abstract: {parsed['abstract']}")

    if parsed.get('body_text'):
        # Truncate body to reasonable length for embedding
        body = parsed['body_text']
        if len(body) > 50000:
            body = body[:50000] + "..."
        parts.append(f"Full text: {body}")

    return '\n\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description='Parse medRxiv JATS XML')
    parser.add_argument('xml_file', nargs='?', help='XML file to parse')
    parser.add_argument('--doi', help='Find and parse by DOI')
    parser.add_argument('--build-index', action='store_true', help='Build DOI index')
    parser.add_argument('--output', choices=['json', 'text', 'embedding'], default='json')
    args = parser.parse_args()

    if args.build_index:
        index = build_doi_index(XML_DIR)
        with open(INDEX_FILE, 'w') as f:
            json.dump(index, f)
        print(f"Saved index with {len(index)} DOIs to {INDEX_FILE}")
        return

    # Load index if exists
    index = {}
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            index = json.load(f)

    # Find file to parse
    xml_path = None
    if args.doi:
        xml_path = find_xml_by_doi(args.doi, index)
        if not xml_path:
            print(f"No XML found for DOI: {args.doi}")
            return
    elif args.xml_file:
        xml_path = Path(args.xml_file)
    else:
        parser.print_help()
        return

    # Parse
    parsed = parse_jats_xml(xml_path)

    if args.output == 'json':
        print(json.dumps(parsed, indent=2))
    elif args.output == 'text':
        print(get_full_text_for_embedding(parsed))
    elif args.output == 'embedding':
        text = get_full_text_for_embedding(parsed)
        print(f"Length: {len(text)} chars")
        print(text[:2000] + "..." if len(text) > 2000 else text)


if __name__ == '__main__':
    main()
