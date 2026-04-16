from typing import List, Dict

def normalize_blocks(blocks: List[Dict]) -> List[Dict]:
    blocks = _merge_equation_blocks(blocks)
    filtered = []
    for b in blocks:
        if b.get('type') in ('equation_block', 'formula') and not b.get('latex') and not b.get('content'):
            continue
        if b.get('type') == 'paragraph' and not b.get('content') and not b.get('formulas'):
            continue
        filtered.append(b)
    return filtered

def _merge_equation_blocks(blocks: List[Dict]) -> List[Dict]:
    new = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        # paragraph → formula → formula_label
        if (i+2 < len(blocks) and
            blocks[i].get('type') == 'paragraph' and
            blocks[i+1].get('type') == 'formula' and
            blocks[i+2].get('type') == 'formula_label'):
            formula = blocks[i+1]
            latex = formula.get('latex')
            if latex:
                eq = {
                    'type': 'equation_block',
                    'context_before': blocks[i].get('content', ''),
                    'latex': latex,
                    'raw': formula.get('raw'),
                    'source': formula.get('source'),
                    'confidence': formula.get('confidence', 0.5),
                    'ast': formula.get('ast'),
                    'label': blocks[i+2].get('content', ''),
                    'section': blocks[i].get('section'),
                    'formulas': [],
                    'semantic_description': blocks[i].get('content', '')[:300]
                }
                if i+3 < len(blocks) and blocks[i+3].get('type') == 'paragraph':
                    eq['description'] = blocks[i+3].get('content')
                    i += 4
                else:
                    i += 3
                new.append(eq)
            else:
                i += 3
            continue
        # одиночная формула
        if b.get('type') == 'formula' and b.get('latex'):
            new.append({
                'type': 'equation_block',
                'latex': b.get('latex'),
                'raw': b.get('raw'),
                'source': b.get('source'),
                'confidence': b.get('confidence', 0.5),
                'ast': b.get('ast'),
                'section': b.get('section'),
                'formulas': [],
                'semantic_description': ''
            })
            i += 1
            continue
        # параграф с формулами внутри – оставляем как есть
        if b.get('type') == 'paragraph' and b.get('formulas'):
            new.append(b)
            i += 1
            continue
        new.append(b)
        i += 1
    return new