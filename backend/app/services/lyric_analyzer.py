import re
from typing import List, Dict, Any
from difflib import SequenceMatcher

# модуль структурного анализа песни, выделение куплетов и припевов
class LyricAnalyzer:

    def __init__(self):
        pass

    # очистка строки от мусорных повторов без удаления пунктуации
    @classmethod
    def _clean_line(cls, text: str) -> str:
        if not text:
            return ""
        t = text.strip()

        # удаление повторяющихся одинаковых слов подряд
        t = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', t, flags=re.IGNORECASE)
        t = re.sub(r'\.{2,}', '...', t)
        t = re.sub(r'\s+', ' ', t).strip()

        if re.fullmatch(r'[\s\.\,\-\:\;\!\?]+', t):
            return ""
        return t

    # разбиение на короткие стихотворные строки
    @classmethod
    def _split_into_poetic_lines(cls, text: str) -> List[str]:
        cleaned = cls._clean_line(text)
        if not cleaned:
            return []

        # разбиение по переводам строк
        raw_parts = re.split(r'[\r\n]+', cleaned)
        res = []
        for p in raw_parts:
            s = p.strip()
            if not s:
                continue
            # разделение составных фраз по пунктуации или регистру
            sub_chunks = re.split(r'[:;]\s*|(?<=[,\.?!])\s+(?=[A-ZА-ЯЁ\u00C0-\u024F])|(?<=[a-zа-яё0-9])\s+(?=[A-ZА-ЯЁ\u00C0-\u024F])', s)
            for chunk in sub_chunks:
                c = chunk.strip()
                if c and len(c) > 1 and not re.fullmatch(r'[\s\.\,\-\:\;\!\?]+', c):
                    res.append(c)

        return res if res else [cleaned]

    # нормализация строки для сравнения без пунктуации
    @classmethod
    def _normalize_for_compare(cls, text: str) -> str:
        t = text.lower()
        t = re.sub(r'[^\w\s]', '', t)
        return re.sub(r'\s+', ' ', t).strip()

    @classmethod
    def _similar(cls, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    # анализ структуры песни на основе естественных пауз и повторяющихся строф
    def analyze_structure(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []

        lines_pool = []
        for s in segments:
            split_lines = self._split_into_poetic_lines(s.get("text", ""))
            for sl in split_lines:
                lines_pool.append({
                    "time_str": s.get("start_str", "00:00"),
                    "text": sl,
                    "start": s.get("start", 0.0),
                    "end": s.get("end", 0.0)
                })

        if not lines_pool:
            return []

        # группировка строк в строфы по паузам между фразами или размеру
        stanzas = []
        cur_stanza = []

        for i, item in enumerate(lines_pool):
            if cur_stanza:
                prev = cur_stanza[-1]
                gap = item["start"] - prev["end"]
                # пауза более 3 секунд или достижение 7-8 строк означает переход к новой строфе
                if gap >= 3.0 or len(cur_stanza) >= 8:
                    stanzas.append(cur_stanza)
                    cur_stanza = []
            cur_stanza.append(item)

        if cur_stanza:
            stanzas.append(cur_stanza)

        if not stanzas:
            stanzas = [lines_pool]

        # определение схожести строф для выявления припевов
        def stanza_text(st):
            return " ".join(self._normalize_for_compare(l["text"]) for l in st)

        is_chorus = [False] * len(stanzas)
        for i in range(len(stanzas)):
            for j in range(i + 1, len(stanzas)):
                txt_i = stanza_text(stanzas[i])
                txt_j = stanza_text(stanzas[j])
                if len(txt_i) >= 15 and len(txt_j) >= 15:
                    sim = self._similar(txt_i, txt_j)
                    if sim >= 0.45:
                        is_chorus[i] = True
                        is_chorus[j] = True

        blocks = []
        verse_idx = 1
        chorus_idx = 1

        for idx, st in enumerate(stanzas):
            if is_chorus[idx]:
                title = f"Припев {chorus_idx}"
                chorus_idx += 1
                b_type = "chorus"
            else:
                title = f"Куплет {verse_idx}"
                verse_idx += 1
                b_type = "verse"

            blocks.append({
                "id": len(blocks) + 1,
                "type": b_type,
                "title": title,
                "lines": st
            })

        return blocks
