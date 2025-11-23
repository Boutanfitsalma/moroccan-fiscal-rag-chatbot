import re
import json
import tempfile
import os
from typing import List, Dict, Tuple, Any, Optional
import fitz  # PyMuPDF
import warnings
import pdfplumber
import easyocr
import camelot
import numpy as np
import cv2
warnings.filterwarnings("ignore")

class FiscalNoteChunker:
    def __init__(self):
        # --- Expressions régulières strictes ---
        self.letter_subsubsection_strict = re.compile(r'^([A-Z])(?:[.\-])\s+(.+)$')
        self.numeric_subsection_strict = re.compile(r'^(\d+)(?:[.\-])\s+(.+)$')
        self.roman_section_pattern = re.compile(r'^([IVX]{1,4})[\s\-\.]+(.+)$')

        # Expressions régulières pour les motifs spéciaux
        self.regex_patterns = {
            'main_title': re.compile(r'^#+\s*NOTE\s+CIRCULAIRE\s+N°\s*\d+', re.IGNORECASE),
            'roman_section': self.roman_section_pattern,
            'numeric_subsection': self.numeric_subsection_strict,
            'letter_subsubsection': self.letter_subsubsection_strict,
            'before_lf': re.compile(r'Avant (?:l\'entrée en vigueur de la LF|la LF)\s+\d{4}', re.IGNORECASE),
            'modifications': re.compile(r'Modifications introduites par la LF \d{4}', re.IGNORECASE),
            'effective_date': re.compile(r'Date d\'effet\s*:', re.IGNORECASE),
            'note': re.compile(r'N\.?B\s*:', re.IGNORECASE),
            'examples': re.compile(r'Exemples? d\'illustration\s*:|Exemple n° \d+\s*:', re.IGNORECASE),
            'summary': re.compile(r'^SOMMAIRE$', re.IGNORECASE),
            'preamble': re.compile(r'^PREAMBULE$', re.IGNORECASE)
        }

        # Variables pour suivre le contexte hiérarchique actif
        self.active_roman_section_title: Optional[str] = None
        self.active_numeric_subsection_title: Optional[str] = None
        self.active_letter_subsubsection_title: Optional[str] = None

        # Tracking des tableaux assignés pour éviter les duplications
        self.assigned_tables: set = set()  # Set de tuples (page_num, table_index)

    def _create_hierarchical_id(self, page_num, roman_sec, num_subsec, letter_subsubsec):
        """Crée un ID hiérarchique court."""
        base_id = f"2025"  # Année du document
        
        if roman_sec:
            # Extraire juste le numéro romain de "I-MESURES SPECIFIQUES..."
            roman_num = roman_sec.split('-')[0].strip()
            base_id += f".{roman_num}"
        
        if num_subsec:
            # Extraire juste le numéro de "1-Augmentation des dotations..."
            num = num_subsec.split('-')[0].strip()
            base_id += f".{num}"
        
        if letter_subsubsec:
            # Extraire juste la lettre de "A-Rappel de l'évolution..."
            letter = letter_subsubsec.split('-')[0].strip()
            base_id += f".{letter}"
        
        # Ajouter le numéro de page
        base_id += f".p{page_num}"
        
        return base_id
    
    def _clean_text(self, x: str) -> str:
        if x is None:
            return ""
        x = str(x)
        x = x.replace('\u00A0', ' ')
        x = re.sub(r'[\u2022\u2023\u25AA\u25CF\u2043\u25E6\uf0b7]', ' ', x)
        x = re.sub(r'\s+', ' ', x).strip()
        return x

    def _normalize_headers(self, headers: list) -> list:
        cleaned = [self._clean_text(h) for h in headers]
        if len(cleaned) == 1:
            parts = re.split(r'\s{2,}|\s*\|\s*|\s*/\s*|\n+', cleaned[0])
            parts = [p for p in (self._clean_text(p) for p in parts) if p]
            if len(parts) >= 2:
                cleaned = parts
        return [h for h in cleaned if h]

    def _normalize_row(self, row: list, headers: list) -> dict:
        out = {}
        for i, h in enumerate(headers):
            val = self._clean_text(row[i] if i < len(row) else "")
            out[h] = val
        return out

    def _table_to_text(self, headers: list, rows: list) -> str:
        lines = []
        for r in rows:
            parts = [f"{h}: {r.get(h, '')}" for h in headers]
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def _normalize_structured_table(self, table: dict) -> Optional[dict]:
        headers = table.get("headers", [])
        data_rows = table.get("data", [])
        if not headers or not data_rows:
            return None
        
        # Vérifier si les données sont déjà des dictionnaires (depuis _clean_camelot_table)
        if data_rows and isinstance(data_rows[0], dict):
            # Données déjà nettoyées par _clean_camelot_table
            normalized_rows = data_rows
        else:
            # Données brutes - appliquer la normalisation
            cleaned_headers = self._normalize_headers(headers)
            normalized_rows = [self._normalize_row(r, cleaned_headers) for r in data_rows]
            headers = cleaned_headers
            
        return {
            "page": table.get("page"),
            "table_index": table.get("table_index"),
            "headers": headers,
            "rows": normalized_rows,
            "as_text": self._table_to_text(headers, normalized_rows)
        }

    def _clean_camelot_table(self, camelot_table):
        """Applique ta fonction clean_table directement sur un objet Camelot."""
        import tempfile
        import os
        
        # Créer un fichier temporaire pour exporter en JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
            temp_path = temp_file.name
            camelot_table.to_json(temp_path)
        
        try:
            # Lire et nettoyer le JSON
            with open(temp_path, "r", encoding="utf-8") as f:
                camelot_json = f.read()
            
            # Appliquer ta logique de nettoyage
            rows = json.loads(camelot_json)
            
            # La première ligne = en-têtes
            headers = [re.sub(r'\s+', ' ', rows[0][str(i)].strip()) for i in range(len(rows[0]))]
            
            clean_rows = []
            for row in rows[1:]:
                clean_row = {}
                for i, header in enumerate(headers):
                    val = row.get(str(i), "").strip()
                    # Nettoyage puces & retours à la ligne
                    val = re.sub(r'[\u2022\u2023\u25AA\u25CF\u2219\u2043\u2219\u25E6\uf0b7]', '', val)  
                    val = re.sub(r'\s+', ' ', val)
                    clean_row[header] = val
                clean_rows.append(clean_row)
            
            return headers, clean_rows
            
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.remove(temp_path)
            except:
                pass

    def _is_valid_roman_section(self, line_stripped: str, roman_num: str, title_text: str) -> bool:
        # ===== CRITÈRES D'EXCLUSION (FAUX POSITIFS) =====
        law_references = [
            r'du CGI', r'de l\'article', r'des articles', r'au paragraphe',
            r'du paragraphe', r'de l\'alinéa', r'et [IVX]+\)', r'[IVX]+ et [IVX]+\)',
            r'[IVX]+\) en vigueur', r'[IVX]+\) du', r'[IVX]+\) de'
        ]
        
        for pattern in law_references:
            if re.search(pattern, line_stripped, re.IGNORECASE):
                return False
        
        if not re.match(r'^' + re.escape(roman_num) + r'[\s\-\.]', line_stripped):
            return False
        
        if not line_stripped.startswith(roman_num):
            return False
        
        if len(roman_num) > 4:
            return False
        
        if re.match(rf'^{re.escape(roman_num)}\)', line_stripped):
            return False
        
        # ===== CRITÈRES D'INCLUSION (VRAIS POSITIFS) =====
        section_keywords = [
            r'MESURES SPECIFIQUES', r'MESURES COMMUNES', r'IMPOT SUR LES SOCIETES',
            r'IMPOT SUR LE REVENU', r'TAXE SUR LA VALEUR AJOUTEE', r'DROITS DE TIMBRE',
            r'DROITS D\'ENREGISTREMENT', r'TAXE SPECIALE'
        ]
        
        for keyword in section_keywords:
            if re.search(keyword, title_text, re.IGNORECASE):
                return True
        
        if len(title_text.strip()) >= 15 and len(title_text.strip()) <= 200:
            if not re.search(r'^(du|de la|de l\'|des|le|la|les|un|une|dans|sur|avec|selon|pour)', title_text.strip(), re.IGNORECASE):
                uppercase_ratio = sum(1 for c in title_text if c.isupper()) / len(title_text) if title_text else 0
                if uppercase_ratio > 0.3:
                    return True
        
        return False

    def _validate_roman_section(self, line_stripped: str) -> tuple:
        match = self.roman_section_pattern.search(line_stripped)
        if not match:
            return False, None, None
        
        roman_num = match.group(1)
        title_text = match.group(2).strip()
        
        is_valid = self._is_valid_roman_section(line_stripped, roman_num, title_text)
        
        if is_valid:
            return True, roman_num, title_text
        else:
            return False, None, None

    
    

    def extract_text_and_tables_from_pdf_with_pages(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extrait le texte page par page. Force l'OCR sur la dernière page."""
        pages_data = []
        stats = {"success": False, "method": None, "total_pages": 0, "total_text_length": 0}

        doc = fitz.open(pdf_path)  # Ouvrir une fois
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                # Pages 1 à n-1 avec pdfplumber
                for page_num, page in enumerate(pdf.pages[:-1], start=1):
                    text_page = page.extract_text() or ""
                    text_page = re.sub(r'[ \u00A0]+', ' ', text_page)
                    pages_data.append({
                        "page_number": page_num,
                        "text": text_page,
                        "tables": []
                    })

                # 🔥 FORCER OCR SUR DERNIÈRE PAGE

                last_page_index = total_pages - 1
                try:
                    # ✅ Appelle la méthode dédiée (reader sera créé à l'intérieur)
                    ocr_text = self._extract_page_with_ocr_reader(doc, last_page_index)
                    ocr_text = re.sub(r'[ \u00A0]+', ' ', ocr_text.strip())
                except Exception as e:
                    print(f"🔥 OCR échoué: {e}")
                    ocr_text = ""

                # Fallback
                if not ocr_text.strip():
                    print("🔥 OCR échoué, fallback sur pdfplumber")
                    ocr_text = pdf.pages[-1].extract_text() or ""
                    ocr_text = re.sub(r'[ \u00A0]+', ' ', ocr_text)

                pages_data.append({
                    "page_number": total_pages,
                    "text": ocr_text,
                    "tables": []
                })
            # Camelot pour les tableaux
            try:
                camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor="lattice")
                for idx, camelot_table in enumerate(camelot_tables):
                    if len(camelot_table.df) > 1:
                        headers, clean_data = self._clean_camelot_table(camelot_table)
                        page_num = camelot_table.page
                        for page_data in pages_data:
                            if page_data['page_number'] == page_num:
                                page_data['tables'].append({
                                    "headers": headers,
                                    "data": clean_data,
                                    "page": page_num,
                                    "table_index": idx
                                })
                                break
            except Exception as e:
                print(f"⚠️ Tableaux non extraits: {e}")

            # Stats
            total_length = sum(len(p['text']) for p in pages_data)
            if pages_data and total_length > 1000:
                stats.update({
                    "success": True,
                    "method": "pdfplumber_with_ocr_last_page",
                    "total_pages": len(pages_data),
                    "total_text_length": total_length,
                    "total_tables": sum(len(p['tables']) for p in pages_data)
                })

        except Exception as e:
            stats["error"] = f"Erreur: {str(e)}"
        finally:
            doc.close()

        return pages_data, stats

    def _extract_page_with_ocr(self, doc, page_index: int) -> str:
        try:
            reader = easyocr.Reader(["fr", "en"])
            return self._extract_page_with_ocr_reader(doc, page_index, reader)
        except Exception as e:
            print(f"OCR failed: {e}")
            return ""

    

    def _extract_page_with_ocr_reader(self, doc, page_index: int, reader=None) -> str:
        try:
            # ✅ Créer le reader si non fourni
            if reader is None:
                reader = easyocr.Reader(["fr", "en"])
            
            print(f"🔥 OCR: Lecture directe sans fichier (page {page_index + 1})")
            mat = fitz.Matrix(3.0, 3.0)  # x3 zoom
            pix = doc[page_index].get_pixmap(matrix=mat)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            else:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
            # ✅ Seuillage adaptatif (meilleur que THRESH_OTSU)
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            ocr_result = reader.readtext(thresh, detail=0, paragraph=False)
            final_text = "\n".join(ocr_result).strip()
            print(f"🔥 OCR: Texte extrait: {repr(final_text[:100])}")
            return final_text

        except Exception as e:
            print(f"🔥 OCR: Échec: {e}")
            return ""
    
    def _find_page_number(self, char_position: int, page_boundaries: List[Dict]) -> Optional[int]:
        if char_position < 0 or not page_boundaries:
            return None
        if char_position >= page_boundaries[-1]['start']:
            return page_boundaries[-2]['page_num'] if len(page_boundaries) > 1 else None

        for i in range(len(page_boundaries) - 1):
            if page_boundaries[i]['start'] <= char_position < page_boundaries[i+1]['start']:
                return page_boundaries[i]['page_num']
        return None

    def _detect_first_section_robustly(self, full_text: str, summary_items: List[str], search_start: int) -> Optional[int]:
        # Stratégie 1: Recherche directe des titres du sommaire avec variations
        if summary_items:
            first_summary_item = summary_items[0]
            base_title = first_summary_item
            
            if base_title.endswith(':'):
                base_title = base_title[:-1]
            if base_title.endswith(' :'):
                base_title = base_title[:-2]
                
            variants = [
                first_summary_item, base_title, base_title + ' :', base_title + ':',
                first_summary_item.replace('- ', '. '), first_summary_item.replace('- ', '.- '),
                first_summary_item.replace(' :', ''),
                re.sub(r'^([IVX]+)[\-\.]', r'\1.-', first_summary_item),
                re.sub(r'^([IVX]+)[\-\.]', r'\1.', first_summary_item),
                re.sub(r'^([IVX]+)[\-\.]', r'\1-', first_summary_item),
            ]
            
            additional_variants = []
            for variant in variants:
                cleaned = re.sub(r'\s*\([^)]+\)', '', variant)
                if cleaned != variant:
                    additional_variants.append(cleaned)
            variants.extend(additional_variants)
            
            variants = list(dict.fromkeys(variants))
            
            for variant in variants:
                pos = full_text.find(variant, search_start)
                if pos != -1:
                    return pos
        
        # Stratégie 2: Recherche par pattern regex flexible
        pattern = re.compile(r'^([IVX]{1,4})[\s\-\.]+.*(?:MESURES|IMPOT|TAXE)', re.MULTILINE | re.IGNORECASE)
        search_text = full_text[search_start:]
        match = pattern.search(search_text)
        if match:
            return search_start + match.start()
        
        # Stratégie 3: Recherche de mots-clés spécifiques
        keywords = [
            "MESURES SPECIFIQUES A L'IMPOT SUR LES SOCIETES",
            "IMPOT SUR LES SOCIETES",
            "MESURES SPECIFIQUES",
        ]
        
        for keyword in keywords:
            pos = full_text.find(keyword, search_start)
            if pos != -1:
                line_start = full_text.rfind('\n', search_start, pos)
                if line_start == -1:
                    line_start = search_start
                else:
                    line_start += 1
                    
                line_text = full_text[line_start:pos + len(keyword)]
                if re.match(r'^\s*[IVX]{1,4}[\s\-\.]', line_text):
                    return line_start
        
        # Stratégie 4: Recherche de la première ligne commençant par un chiffre romain
        lines = full_text[search_start:].split('\n')
        current_pos = search_start
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line and re.match(r'^[IVX]{1,4}[\s\-\.]', stripped_line):
                return current_pos
            current_pos += len(line) + 1
        
        return None
    
    def _detect_document_structure(self, full_text: str) -> Dict[str, Any]:
        structure = {
            'summary_start': None,
            'preamble_start_in_summary': None,
            'preamble_content_start': None,
            'first_section_start': None,
            'summary_items': []
        }

        try:
            # 1. Trouver SOMMAIRE
            pos_summary = full_text.find("SOMMAIRE")
            if pos_summary == -1:
                return structure

            structure['summary_start'] = pos_summary

            # 2. Trouver PREAMBULE dans le sommaire
            pos_preamble_in_summary = full_text.find("PREAMBULE", pos_summary + len("SOMMAIRE"))
            if pos_preamble_in_summary != -1:
                structure['preamble_start_in_summary'] = pos_preamble_in_summary

            # 3. Délimitation de la zone de recherche du contenu du sommaire
            summary_search_start_pos = pos_summary + len("SOMMAIRE")
            summary_search_start_pos = full_text.find('\n', summary_search_start_pos)
            if summary_search_start_pos == -1:
                summary_search_start_pos = pos_summary + len("SOMMAIRE")
            else:
                summary_search_start_pos += 1

            estimated_summary_end = pos_summary + 6000
            
            # 4. Extraction préliminaire du contenu du sommaire
            summary_content_text = full_text[summary_search_start_pos:estimated_summary_end]
            summary_lines = summary_content_text.splitlines()
            
            for line in summary_lines:
                cleaned_line = line.strip()
                if not cleaned_line or cleaned_line.upper() in ["SOMMAIRE", "PREAMBULE"]:
                    continue
                
                match = re.search(r'^([IVXLCDM]+)[\s\-\.]+(.+)', cleaned_line)
                if match and (len(match.group(1)) > 1 or match.group(1) in 'IVX'):
                    roman_num = match.group(1)
                    title_text = match.group(2).strip()
                    separator = '-' if '- ' in cleaned_line else '.'
                    section_title = f"{roman_num}{separator} {title_text}"
                    structure['summary_items'].append(section_title)

            # 5. Détection robuste du préambule
            temp_preamble_content_start = None
            
            preamble_indicators = [
                "La Loi de Finances pour l'année budgétaire",
                "Les mesures fiscales introduites par la loi de fin",
                "Dans le cadre des réformes structurelles",
                "Le Gouvernement poursuit le processus",
                "La présente note circulaire a pour objet",
                "Cette réforme vise",
                "Dans le cadre de la continuité"
            ]
            
            search_start = pos_preamble_in_summary + len("PREAMBULE") if pos_preamble_in_summary != -1 else pos_summary + 2000
            
            for indicator in preamble_indicators:
                pos = full_text.find(indicator, search_start)
                if pos != -1 and pos < search_start + 4000: 
                    temp_preamble_content_start = pos
                    break
            
            if not temp_preamble_content_start:
                first_preambule = full_text.find("PREAMBULE", pos_summary)
                if first_preambule != -1:
                    second_preambule = full_text.find("PREAMBULE", first_preambule + len("PREAMBULE"))
                    if (second_preambule != -1 and 
                        second_preambule > first_preambule + 500 and 
                        second_preambule > pos_summary):
                        preamble_line_end = full_text.find('\n', second_preambule)
                        if preamble_line_end != -1:
                            temp_preamble_content_start = preamble_line_end + 1
            
            structure['preamble_content_start'] = temp_preamble_content_start

            # 6. Détection robuste de la première section
            search_start_for_section = temp_preamble_content_start if temp_preamble_content_start else pos_summary + 3000
            
            first_section_pos = self._detect_first_section_robustly(
                full_text, 
                structure['summary_items'], 
                search_start_for_section
            )
            
            structure['first_section_start'] = first_section_pos

        except Exception:
            pass

        return structure

    def process_document(self, pages_data: List[Dict[str, Any]]) -> List[Dict]:
        if not pages_data:
            return []

        # Réinitialiser le tracking des tableaux assignés pour ce document
        self.assigned_tables.clear()

        full_text = "\n".join(page['text'] for page in pages_data)
        print(f"🔥 DEBUG: Dernières 500 chars du full_text:")
        print(repr(full_text[-500:]))  
        page_boundaries = []
        cumulative_length = 0
        for page in pages_data:
            page_boundaries.append({'start': cumulative_length, 'page_num': page['page_number']})
            cumulative_length += len(page['text']) + 1
        page_boundaries.append({'start': cumulative_length, 'page_num': None})

        chunks = []

        structure = self._detect_document_structure(full_text)
        
        if not isinstance(structure, dict):
            return chunks

        current_pos = 0

        # --- Chunk: Avant SOMMAIRE ---
        if structure.get('summary_start') is not None and structure['summary_start'] > 0:
            content = full_text[current_pos:structure['summary_start']].strip()
            if content:
                chunk = self._create_chunk_with_pages(
                    content, 
                    {'type': 'document_header'}, 
                    current_pos, structure['summary_start'], 
                    page_boundaries
                )
                table_metadata = self._enrich_chunk_with_tables(
                    content,
                    chunk['metadata'].get('page_start', 1),
                    chunk['metadata'].get('page_end'),
                    pages_data,
                    current_pos,
                    full_text
                )
                chunk['metadata'].update(table_metadata)
                chunks.append(chunk)
                current_pos = structure['summary_start']

        # --- Chunk: SOMMAIRE ---
        if structure.get('summary_start') is not None:
            summary_end_pos = structure.get('preamble_content_start') or structure.get('first_section_start') or len(full_text)
            content = full_text[structure['summary_start']:summary_end_pos].strip()
            if content:
                chunk = self._create_chunk_with_pages(
                    content, 
                    {'type': 'summary'}, 
                    structure['summary_start'], summary_end_pos, 
                    page_boundaries
                )
                table_metadata = self._enrich_chunk_with_tables(
                    content,
                    chunk['metadata'].get('page_start', 1),
                    chunk['metadata'].get('page_end'),
                    pages_data,
                    structure['summary_start'],
                    full_text
                )
                chunk['metadata'].update(table_metadata)
                chunks.append(chunk)
                current_pos = summary_end_pos

        # --- Chunk: PREAMBULE ---
        if structure.get('preamble_content_start') is not None:
            preamble_end_pos = structure.get('first_section_start') or len(full_text)
            content = full_text[structure['preamble_content_start']:preamble_end_pos].strip()
            if content:
                chunk = self._create_chunk_with_pages(
                    content, 
                    {'type': 'preamble'}, 
                    structure['preamble_content_start'], preamble_end_pos, 
                    page_boundaries
                )
                table_metadata = self._enrich_chunk_with_tables(
                    content,
                    chunk['metadata'].get('page_start', 1),
                    chunk['metadata'].get('page_end'),
                    pages_data,
                    structure['preamble_content_start'],
                    full_text
                )
                chunk['metadata'].update(table_metadata)
                chunks.append(chunk)
                current_pos = preamble_end_pos

        # --- Réinitialiser le contexte hiérarchique ---
        self.active_roman_section_title = None
        self.active_numeric_subsection_title = None
        self.active_letter_subsubsection_title = None

        # --- Chunks: Reste du document ---
        if structure.get('first_section_start') is not None and structure['first_section_start'] < len(full_text):
            rest_of_text = full_text[structure['first_section_start']:]
            offset_in_full_text = structure['first_section_start']
            chunks.extend(self._chunk_rest_of_document_with_tables(rest_of_text, page_boundaries, full_text, offset_in_full_text, pages_data))
        elif current_pos < len(full_text):
            rest_of_text = full_text[current_pos:]
            chunks.extend(self._chunk_rest_of_document_with_tables(rest_of_text, page_boundaries, full_text, current_pos, pages_data))

        return chunks

    def _create_chunk_with_pages(self, content: str, metadata: Dict, start_pos: int, end_pos: int, page_boundaries: List[Dict]) -> Dict:
        page_start = self._find_page_number(start_pos, page_boundaries)
        page_end = self._find_page_number(end_pos, page_boundaries)
        
        if page_start:
            metadata['page_start'] = page_start
        if page_end and page_end != page_start:
            metadata['page_end'] = page_end
            
        return self._create_chunk(content, metadata)

    def _chunk_rest_of_document(self, text: str, page_boundaries: List[Dict], full_text: str, offset_in_full_text: int) -> List[Dict]:
        lines = text.splitlines()
        chunks = []
        current_chunk_lines = []
        current_metadata = {}
        
        current_chunk_start_pos_in_full = None
        current_chunk_end_pos_in_full = None

        def finalize_and_add_chunk():
            nonlocal current_chunk_lines, current_metadata, chunks
            nonlocal current_chunk_start_pos_in_full, current_chunk_end_pos_in_full

            if not current_chunk_lines:
                return

            content = '\n'.join(current_chunk_lines).strip()
            if not content:
                current_chunk_lines = []
                current_metadata = {}
                current_chunk_start_pos_in_full = None
                current_chunk_end_pos_in_full = None
                return

            page_num_start = self._find_page_number(current_chunk_start_pos_in_full, page_boundaries) if current_chunk_start_pos_in_full is not None else None
            page_num_end = self._find_page_number(current_chunk_end_pos_in_full, page_boundaries) if current_chunk_end_pos_in_full is not None else None

            final_metadata = {}
            if self.active_roman_section_title:
                final_metadata['roman_section_title'] = self.active_roman_section_title
            if self.active_numeric_subsection_title:
                final_metadata['numeric_subsection_title'] = self.active_numeric_subsection_title
            if self.active_letter_subsubsection_title:
                final_metadata['letter_subsubsection_title'] = self.active_letter_subsubsection_title
            
            final_metadata.update(current_metadata)
            
            if page_num_start:
                final_metadata['page_start'] = page_num_start
            if page_num_end and page_num_end != page_num_start:
                final_metadata['page_end'] = page_num_end

            chunks.append(self._create_chunk(content, final_metadata))
            
            current_chunk_lines = []
            current_metadata = {}
            current_chunk_start_pos_in_full = None
            current_chunk_end_pos_in_full = None

        def merge_title_lines(start_line_index: int, initial_title: str) -> tuple:
            titre_complet = initial_title
            lignes_fusionnees = 0
            
            next_line_index = start_line_index + 1
            
            while next_line_index < len(lines):
                next_line = lines[next_line_index].strip()
                
                if not next_line:
                    next_line_index += 1
                    continue
                
                if next_line[0].islower():
                    titre_complet += " " + next_line
                    lignes_fusionnees += 1
                    next_line_index += 1
                else:
                    break
            
            return titre_complet, lignes_fusionnees

        current_pos_in_text = 0
        i = 0
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.rstrip()
            line_start_in_text = current_pos_in_text
            line_end_in_text = line_start_in_text + len(line)
            
            if not line_stripped:
                if current_chunk_lines:
                    current_chunk_lines.append(line)
                    current_chunk_end_pos_in_full = offset_in_full_text + line_end_in_text
                current_pos_in_text = line_end_in_text + 1
                i += 1
                continue

            current_chunk_end_pos_in_full = offset_in_full_text + line_end_in_text

            is_new_section_header = False
            lines_to_skip = 0
            
            # 1. Vérifier les sections romaines avec validation stricte
            is_valid_roman, roman_num, section_title_text = self._validate_roman_section(line_stripped)
            if is_valid_roman:
                finalize_and_add_chunk()
                is_new_section_header = True
                
                titre_complet, lignes_fusionnees = merge_title_lines(i, line_stripped)
                lines_to_skip = lignes_fusionnees
                
                match = self.roman_section_pattern.search(titre_complet)
                if match:
                    roman_num = match.group(1)
                    section_title_text = match.group(2).strip()
                
                self.active_roman_section_title = f"{roman_num}-{section_title_text}"
                self.active_numeric_subsection_title = None
                self.active_letter_subsubsection_title = None
                current_metadata = {
                    'type': 'roman_section',
                    'section': self.active_roman_section_title
                }
                current_chunk_lines.append(titre_complet)
                current_chunk_start_pos_in_full = offset_in_full_text + line_start_in_text

            # 2. Vérifier les sous-sections numériques
            elif self.active_roman_section_title and (match := self.regex_patterns['numeric_subsection'].search(line_stripped)):
                finalize_and_add_chunk()
                is_new_section_header = True
                
                titre_complet, lignes_fusionnees = merge_title_lines(i, line_stripped)
                lines_to_skip = lignes_fusionnees
                
                match_complet = self.regex_patterns['numeric_subsection'].search(titre_complet)
                if match_complet:
                    num_val = match_complet.group(1)
                    subsection_title_text = match_complet.group(2).strip()
                else:
                    num_val = match.group(1)
                    subsection_title_text = titre_complet[len(num_val)+1:].strip()
                
                self.active_numeric_subsection_title = f"{num_val}-{subsection_title_text}"
                self.active_letter_subsubsection_title = None
                current_metadata = {
                    'type': 'numeric_subsection',
                    'subsection': self.active_numeric_subsection_title
                }
                current_chunk_lines.append(titre_complet)
                current_chunk_start_pos_in_full = offset_in_full_text + line_start_in_text

            # 3. Vérifier les sous-sous-sections lettrées
            elif self.active_numeric_subsection_title and (match := self.regex_patterns['letter_subsubsection'].search(line_stripped)):
                finalize_and_add_chunk()
                is_new_section_header = True
                
                titre_complet, lignes_fusionnees = merge_title_lines(i, line_stripped)
                lines_to_skip = lignes_fusionnees
                
                match_complet = self.regex_patterns['letter_subsubsection'].search(titre_complet)
                if match_complet:
                    letter_val = match_complet.group(1)
                    subsubsection_title_text = match_complet.group(2).strip()
                else:
                    letter_val = match.group(1)
                    subsubsection_title_text = titre_complet[len(letter_val)+1:].strip()
                
                self.active_letter_subsubsection_title = f"{letter_val}-{subsubsection_title_text}"
                current_metadata = {
                    'type': 'letter_subsubsection',
                    'subsection': self.active_letter_subsubsection_title
                }
                current_chunk_lines.append(titre_complet)
                current_chunk_start_pos_in_full = offset_in_full_text + line_start_in_text

            if not is_new_section_header:
                if not current_metadata:
                    current_metadata = {'type': 'content_paragraph'}
                    current_chunk_start_pos_in_full = offset_in_full_text + line_start_in_text

                current_chunk_lines.append(line_stripped)

                for pattern_name, pattern in self.regex_patterns.items():
                    if pattern_name in ['before_lf', 'modifications', 'effective_date', 'note', 'examples']:
                        if pattern.search(line_stripped):
                            current_metadata[pattern_name] = True
                            if current_metadata['type'] == 'content_paragraph':
                                current_metadata['type'] = pattern_name
                            break

            current_pos_in_text = line_end_in_text + 1
            i += 1 + lines_to_skip
            
            for skip_idx in range(lines_to_skip):
                if i - lines_to_skip + skip_idx < len(lines):
                    current_pos_in_text += len(lines[i - lines_to_skip + skip_idx]) + 1

        finalize_and_add_chunk()
        return chunks

    def _enrich_chunk_with_tables(self, content: str, page_start: int, page_end: int, all_pages_data: List[Dict], chunk_start_pos_in_full_text: int = None, full_text: str = None) -> Dict:
        tables = []

        # Calculer la position de fin du chunk dans le texte complet
        chunk_end_pos_in_full_text = None
        if chunk_start_pos_in_full_text is not None and full_text is not None:
            chunk_end_pos_in_full_text = chunk_start_pos_in_full_text + len(content)

        for page_data in all_pages_data:
            page_num = page_data['page_number']
            if page_start <= page_num <= (page_end or page_start):
                if 'tables' in page_data:
                    for t in page_data['tables']:
                        # Créer un identifiant unique pour ce tableau
                        table_id = (page_num, t.get("table_index", "unk"))

                        # Vérifier si ce tableau a déjà été assigné à un autre chunk
                        if table_id in self.assigned_tables:
                            continue  # Skip ce tableau, il est déjà assigné

                        # Si on a les informations de position, vérifier si le tableau appartient vraiment à ce chunk
                        if chunk_start_pos_in_full_text is not None and full_text is not None:
                            # Calculer la position estimée du tableau dans le texte complet
                            table_position = self._estimate_table_position_in_text(page_num, all_pages_data, full_text)

                            # Vérifier si le tableau est dans la plage de ce chunk
                            if table_position is not None:
                                if not (chunk_start_pos_in_full_text <= table_position <= chunk_end_pos_in_full_text):
                                    continue  # Le tableau n'est pas dans ce chunk, skip

                        normalized = self._normalize_structured_table({
                            "headers": t["headers"],
                            "data": t["data"],
                            "page": t.get("page", page_num),
                            "table_index": t.get("table_index", "unk")
                        })
                        if normalized:
                            # Marquer ce tableau comme assigné
                            self.assigned_tables.add(table_id)

                            table_info = {
                                "page": normalized["page"],
                                "table_index": normalized["table_index"],
                                "headers": normalized["headers"],
                                "data": normalized["rows"],
                                "as_text": normalized["as_text"],
                                "row_count": len(normalized["rows"]),
                                "col_count": len(normalized["headers"]),
                                "type": "structured_table"
                            }
                            tables.append(table_info)
        
        table_metadata = {}
        if tables:
            table_metadata['has_tables'] = True
            table_metadata['table_count'] = len(tables)
            table_metadata['tables'] = tables
            search_index = {}
            for table in tables:
                for row in table.get("data", []):
                    for key, value in row.items():
                        if key.lower() not in search_index:
                            search_index[key.lower()] = []
                        search_index[key.lower()].append({
                            'value': value,
                            'table_page': table['page'],
                            'table_index': table['table_index']
                        })
            table_metadata['search_index'] = search_index
        
        return table_metadata

    def _estimate_table_position_in_text(self, table_page: int, all_pages_data: List[Dict], full_text: str) -> int:
        """
        Estime la position d'un tableau dans le texte complet en se basant sur sa page.
        Retourne la position estimée au milieu de la page où se trouve le tableau.
        """
        try:
            # Calculer la position de début de la page du tableau
            cumulative_length = 0
            for page_data in all_pages_data:
                if page_data['page_number'] == table_page:
                    # Retourner la position au milieu de cette page
                    page_text_length = len(page_data.get('text', ''))
                    return cumulative_length + (page_text_length // 2)
                cumulative_length += len(page_data.get('text', '')) + 1  # +1 pour le \n entre pages

            # Si la page n'est pas trouvée, retourner None
            return None
        except Exception:
            return None

    def _chunk_rest_of_document_with_tables(self, text: str, page_boundaries: List[Dict], full_text: str, offset_in_full_text: int, all_pages_data: List[Dict]) -> List[Dict]:
        chunks = self._chunk_rest_of_document(text, page_boundaries, full_text, offset_in_full_text)

        for i, chunk in enumerate(chunks):
            page_start = chunk['metadata'].get('page_start')
            page_end = chunk['metadata'].get('page_end', page_start)

            if page_start:
                # Calculer la position approximative du chunk dans le texte complet
                # En utilisant la position relative dans le texte traité + l'offset
                if i == 0:
                    chunk_start_pos_in_full_text = offset_in_full_text
                else:
                    # Estimer la position en additionnant les longueurs des chunks précédents
                    estimated_relative_pos = sum(len(c['content']) for c in chunks[:i]) + i  # +i pour les sauts de ligne
                    chunk_start_pos_in_full_text = offset_in_full_text + estimated_relative_pos

                table_metadata = self._enrich_chunk_with_tables(
                    chunk['content'],
                    page_start,
                    page_end,
                    all_pages_data,
                    chunk_start_pos_in_full_text,
                    full_text
                )
                chunk['metadata'].update(table_metadata)

        return chunks

    def _create_chunk(self, content: str, metadata: Dict) -> Dict:
        # Générer l'ID hiérarchique
        page_num = metadata.get('page_start', 1)
        chunk_id = self._create_hierarchical_id(
            page_num,
            metadata.get('roman_section_title'),
            metadata.get('numeric_subsection_title'),
            metadata.get('letter_subsubsection_title')
        )
        
        # Ajouter l'ID aux métadonnées
        metadata['id'] = chunk_id
        
        return {
            'content': content,
            'metadata': metadata,
            'length': len(content),
            'word_count': len(content.split())
        }

    def save_chunks(self, chunks: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
    def print_chunk_stats(self, chunks: List[Dict]):
        total_chunks = len(chunks)
        total_words = sum(chunk['word_count'] for chunk in chunks)
        avg_words = total_words / total_chunks if total_chunks else 0

        print(f"Nombre total de chunks: {total_chunks}")
        print(f"Nombre total de mots: {total_words}")
        print(f"Moyenne de mots par chunk: {avg_words:.2f}")

        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk['metadata'].get('type', 'unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

        print("\nRépartition par type:")
        for chunk_type, count in sorted(type_counts.items()):
            print(f"  {chunk_type}: {count}")

        print("\nSections et Sous-sections détectées:")
        last_roman = ""
        last_numeric = ""
        for chunk in chunks:
            meta = chunk['metadata']
            roman_title = meta.get('roman_section_title', '')
            numeric_title = meta.get('numeric_subsection_title', '')
            letter_title = meta.get('letter_subsubsection_title', '')
            
            if roman_title and roman_title != last_roman:
                print(f"  Section: {roman_title}")
                last_roman = roman_title
                last_numeric = ""
            
            if numeric_title and numeric_title != last_numeric:
                print(f"    Sous-section: {numeric_title}")
                last_numeric = numeric_title
            
            if letter_title:
                print(f"      Sous-sous-section: {letter_title}")
                 
            pages = f"(Page: {meta.get('page_start', 'N/A')}"
            if meta.get('page_end') and meta['page_end'] != meta.get('page_start'):
                pages += f"-{meta['page_end']}"
            pages += ")"
            
            if meta.get('type') not in ['roman_section', 'numeric_subsection', 'letter_subsubsection', 'summary', 'preamble', 'document_header']:
                content_preview = chunk['content'][:50].replace('\n', ' ')
                print(f"        [{meta.get('type', 'N/A')}] {pages} - {content_preview}...")

