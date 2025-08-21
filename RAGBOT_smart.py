#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGBot v9 — FINAL — Tkinter (Documents + Internet) with Modern Sidebar (India/UK cards)

This file merges:
- Robust RAG engine (Docs + Web), judge, smart final answer, sources window
- Stable two‑pane layout with fixed‑width modern sidebar matching your design
- Avatar header, title “Solomons RAGBOT”, subtitle “AI Assistant”
- Two cards side‑by‑side (India, UK): Date, Big Time (HH:MM:SS), TZ, and Live News links
- Timezone dialog
- 4K friendly, graceful fallbacks for missing images

Run:
  python ragbot_tk_v9_final.py

Asset paths (as requested):
  profile: C:\\Python\\Rag-ChatBot-main_1\\venv\\documents\\profile.jpg
  India flag: C:\\Python\\Rag-ChatBot-main_1\\venv\\documents\\india.jpg
  UK flag:    C:\\Python\\Rag-ChatBot-main_1\\venv\\documents\\uk.png
"""

import os, re, glob, io, json, html, urllib.parse, threading, webbrowser
from datetime import datetime, timezone
from typing import List, Tuple

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False

import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Tk
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

# Timezones
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# -------------------- Config --------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
DOCS_PRIMARY = os.path.join(ROOT, "venv", "documents")
DOCS_FALLBACK = os.path.join(ROOT, "documents")
DOCS_DIR = os.environ.get("DOCUMENTS_DIR", DOCS_PRIMARY if os.path.isdir(DOCS_PRIMARY) else DOCS_FALLBACK)

OWNER = os.environ.get("RAGBOT_OWNER", "Solomon")
APP_TITLE = "Solomons RAGBOT"
APP_SUBTITLE = "AI Assistant"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
TIMEOUT = 15

MEMORY_PATH = os.path.join(ROOT, "ragbot_memory.json")
MEMORY_LIMIT = 50

PROFILE_IMG = os.path.join(DOCS_DIR, "profile.jpg")
INDIA_FLAG = os.path.join(DOCS_DIR, "india.jpg")
UK_FLAG    = os.path.join(DOCS_DIR, "uk.png")

DARK_BG = "#0d1117"
CARD_BG = "#161b22"
TEXT = "#e5e7eb"
SUBTLE = "#9ca3af"
ACCENT = "#60a5fa"

# -------------------- Text utils --------------------
_whitespace_re = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not t: return ""
    t = html.unescape(t)
    t = t.replace("\xa0", " ").replace("\u200b", "")
    t = _whitespace_re.sub(" ", t)
    t = re.sub(r"(?i)copyright.*?all rights reserved.*?$", "", t)
    return t.strip()

def sent_split(text: str):
    text = (text or "").strip()
    text = _whitespace_re.sub(" ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [s.strip() for s in parts if s.strip()]

def soft_wrap(paragraphs: List[str]) -> str:
    text = " ".join([p.strip() for p in paragraphs if p and p.strip()])
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r"\( ", "(", text)
    return text.strip()

def top_sentences(paragraph: str, max_sents: int = 5) -> List[str]:
    sents = sent_split(paragraph)
    if not sents: return []
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    try:
        X = vec.fit_transform(sents)
        scores = (X * X.T).toarray().sum(axis=1)
    except ValueError:
        return sents[:max_sents]
    idx = np.argsort(-scores)[:max_sents]
    ranked = [sents[i] for i in idx]
    order = {s:i for i,s in enumerate(sents)}
    ranked.sort(key=lambda s: order.get(s, 0))
    return ranked

def gentle_paraphrase(s: str) -> str:
    swaps = {
        "however": "but",
        "in addition": "also",
        "additionally": "also",
        "therefore": "so",
        "thus": "so",
        "in other words": "simply put",
        "for example": "for instance",
        "for instance": "for example",
    }
    out = s
    for k, v in swaps.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.I)
    out = re.sub(r"\b(and)\s+\1\b", r"\1", out, flags=re.I)
    return out

_COMMON_TYPO = {
    "waht": "what", "teh": "the", "whcih": "which",
    "recieve": "receive", "definately": "definitely",
    "enviroment": "environment", "seperate": "separate",
    "acheive": "achieve", "occured": "occurred",
}

def normalize_query(q: str) -> str:
    words = q.split()
    fixed = [_COMMON_TYPO.get(w.lower(), w) for w in words]
    return " ".join(fixed)

def detect_intent(q: str) -> str:
    ql = q.strip().lower()
    if re.search(r"^(what is|who is|define|definition of)\b", ql):
        return "define"
    if "compare" in ql or "vs" in ql:
        return "compare"
    if any(k in ql for k in ["how to", "steps", "tutorial"]):
        return "howto"
    return "generic"

# --------- Loaders ---------

def read_pdf_with_pages(path):
    pages = []
    try:
        from PyPDF2 import PdfReader
        pdf = PdfReader(path)
        for i, page in enumerate(pdf.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""
            raw = clean_text(raw)
            if raw:
                pages.append({"source": path, "page": i, "text": raw})
    except Exception:
        pass
    return pages


def read_docx(path):
    try:
        from docx import Document
        doc = Document(path)
        txt = "\n".join(p.text for p in doc.paragraphs)
        return [{"source": path, "page": None, "text": clean_text(txt)}]
    except Exception:
        return []


def read_txt(path):
    try:
        txt = open(path, "r", encoding="utf-8", errors="ignore").read()
        return [{"source": path, "page": None, "text": clean_text(txt)}]
    except Exception:
        return []


def read_csvfile(path):
    try:
        lines = open(path, "r", encoding="utf-8", errors="ignore").read()
        return [{"source": path, "page": None, "text": clean_text(lines)}]
    except Exception:
        return []


def read_xlsx(path):
    if pd is None: return []
    try:
        x = pd.read_excel(path, sheet_name=None)
        frames = []
        for _, df in (x or {}).items():
            frames.append(df.astype(str))
        if not frames:
            return []
        cat = pd.concat(frames)
        txt = "\n".join([", ".join(row) for row in cat.values.tolist()])
        return [{"source": path, "page": None, "text": clean_text(txt)}]
    except Exception:
        return []


def read_image_ocr(path):
    try:
        import pytesseract
        if PIL_OK:
            im = Image.open(path)
            txt = pytesseract.image_to_string(im)
            return [{"source": path, "page": None, "text": clean_text(txt)}]
    except Exception:
        pass
    return []

LOADER_BY_EXT = {
    ".pdf": read_pdf_with_pages,
    ".docx": read_docx,
    ".txt": read_txt,
    ".md": read_txt,
    ".csv": read_csvfile,
    ".xlsx": read_xlsx,
    ".xls": read_xlsx,
    ".png": read_image_ocr,
    ".jpg": read_image_ocr,
    ".jpeg": read_image_ocr,
}

def iter_doc_files(folder):
    for ext in LOADER_BY_EXT.keys():
        for p in glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True):
            yield p


def chunk_text(raw: str, max_chars=1200, overlap=150):
    raw = clean_text(raw)
    if not raw: return []
    chunks, i = [], 0
    while i < len(raw):
        piece = raw[i:i+max_chars]
        last_stop = max(piece.rfind("."), piece.rfind("?"), piece.rfind("!"))
        if last_stop > 400:
            piece = piece[:last_stop+1]
        if len(piece) < 200 and i > 0:
            if chunks:
                chunks[-1] = chunks[-1] + " " + piece
            else:
                chunks.append(piece)
            break
        chunks.append(piece.strip())
        i += max_chars - overlap
    return [c for c in chunks if c]


def load_all_documents(folder: str):
    records = []
    if os.path.isdir(folder):
        for f in iter_doc_files(folder):
            loader = LOADER_BY_EXT.get(os.path.splitext(f)[1].lower())
            if not loader: continue
            try:
                for blk in loader(f):
                    for c in chunk_text(blk["text"]):
                        records.append({"source": f, "page": blk.get("page"), "text": c})
            except Exception:
                continue
    return records

# -------------------- Index + QA --------------------
class TfIdfIndex:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.rows = []
    def fit(self, rows):
        self.rows = rows or []
        texts = [r["text"] for r in self.rows]
        if not texts:
            self.vectorizer, self.matrix = None, None
            return
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_df=0.95)
        self.matrix = self.vectorizer.fit_transform(texts)
    def search(self, query, topk=6):
        if self.vectorizer is None or self.matrix is None or not self.rows:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).ravel()
        idxs = np.argsort(-sims)[:topk]
        out = []
        for i in idxs:
            out.append({**self.rows[i], "score": float(sims[i])})
        return out

DOC_INDEX = TfIdfIndex()

def doc_answer_paragraph(query: str):
    rows = DOC_INDEX.search(query, topk=8)
    rows = [r for r in rows if r["score"] > 0.05]
    if not rows:
        return "—", []
    context = " ".join(r["text"] for r in rows)
    sents = top_sentences(context, max_sents=6)
    sents = [gentle_paraphrase(s) for s in sents]
    sents = [s for s in sents if len(s.split()) >= 6]
    seen = set(); uniq = []
    for s in sents:
        k = s.lower()
        if k in seen: continue
        seen.add(k); uniq.append(s)
    paragraph = soft_wrap(uniq) if uniq else "—"

    cites = []
    seen_keys = set()
    for r in rows:
        key = (r["source"], r.get("page"))
        if key not in seen_keys:
            seen_keys.add(key)
            cites.append(key)
        if len(cites) >= 5:
            break
    return paragraph or "—", cites

# -------------------- Web --------------------

def ddg_search(q: str, num: int = 5):
    params = {"q": q, "kl": "in-en"}
    url = "https://duckduckgo.com/html/"
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if not href:
                continue
            if "uddg=" in href:
                try:
                    actual = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [""])[0]
                except Exception:
                    actual = href
                href = actual or href
            links.append(href)
            if len(links) >= num:
                break
        return links
    except Exception:
        return []


def fetch_text(url: str):
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","header","footer","noscript","form","aside"]):
            tag.decompose()
        blocks = []
        for t in soup.find_all(["article","section","p","li","div"]):
            txt = t.get_text(" ", strip=True)
            txt = clean_text(txt)
            if not txt or len(txt) < 80:
                continue
            if "Archived from the original" in txt or "All rights reserved" in txt:
                continue
            blocks.append(txt)
        joined = " ".join(blocks)
        if len(joined) > 12000:
            joined = joined[:12000]
        return joined
    except Exception:
        return ""


def _is_generic_definition(q: str) -> bool:
    return bool(re.search(r"^(what is|who is|define|definition of)\b", q.strip().lower()))


def _wiki_lead(q: str) -> Tuple[str, str]:
    try:
        topic = q.strip()
        topic = re.sub(r"^(what is|who is|define|definition of)\s+", "", topic, flags=re.I)
        topic = topic.strip().rstrip("?")
        if not topic:
            return "", ""
        api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
        r = requests.get(api, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        if r.status_code == 200:
            j = r.json()
            txt = clean_text((j.get("extract") or "")[:1000])
            if txt:
                page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(j.get("title", topic).replace(" ", "_"))
                return txt, page_url
    except Exception:
        pass
    return "", ""


def web_answer_paragraph(query: str):
    if _is_generic_definition(query):
        lead, page = _wiki_lead(query)
        if lead:
            return lead, [page] if page else []
    urls = ddg_search(query, num=5)
    corpus, used = [], []
    for u in urls:
        t = fetch_text(u)
        if len(t) > 400:
            corpus.append(t); used.append(u)
        if len(corpus) >= 4:
            break
    if not corpus:
        return "—", []
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform(corpus + [query])
    sims = cosine_similarity(X[-1], X[:-1]).ravel()
    top_idx = np.argsort(-sims)[:3]
    context = " ".join(corpus[i] for i in top_idx if sims[i] > 0.05)
    sents = top_sentences(context, max_sents=7)
    sents = [gentle_paraphrase(s) for s in sents]
    sents = [s for s in sents if len(s.split()) >= 6]
    seen = set(); unique = []
    for s in sents:
        key = s.lower()
        if key in seen: continue
        seen.add(key); unique.append(s)
    paragraph = soft_wrap(unique) if unique else "—"
    used_urls = [used[i] for i in top_idx if sims[i] > 0.05]
    return paragraph or "—", used_urls[:5]

# -------------------- Judge & SFA --------------------

def key_terms(text: str, k: int = 6):
    if text == "—": return []
    vec = TfidfVectorizer(ngram_range=(1,1), stop_words="english")
    try:
        X = vec.fit_transform(sent_split(text))
    except ValueError:
        return []
    terms = np.array(vec.get_feature_names_out())
    weights = X.toarray().sum(axis=0)
    idx = np.argsort(-weights)[:k]
    return [t for t in terms[idx] if re.match(r"[A-Za-z][A-Za-z0-9\-]+$", t)][:k]


def relevance_label(q, p):
    if p == "—": return "none"
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform([q, p])
    sim = float(cosine_similarity(X[0], X[1])[0,0])
    if sim >= 0.6: return "high"
    if sim >= 0.35: return "medium"
    return "low"


def judge_line(q: str, doc_p: str, web_p: str) -> str:
    if doc_p == "—" or web_p == "—":
        parts = []
        rd = relevance_label(q, doc_p)
        rw = relevance_label(q, web_p)
        if doc_p != "—": parts.append(f"docs relevance {rd}")
        if web_p != "—": parts.append(f"web relevance {rw}")
        return ("; ".join(parts) + ".") if parts else ""
    d = set(key_terms(doc_p, 6)); w = set(key_terms(web_p, 6))
    agree = d & w
    doc_only = d - w
    web_only = w - d
    a = ", ".join(sorted(list(agree))[:3]) if agree else "core points"
    add_d = next(iter(doc_only), None)
    add_w = next(iter(web_only), None)
    rd = relevance_label(q, doc_p)
    rw = relevance_label(q, web_p)
    bits = [f"Both sides align on {a}"]
    if add_d: bits.append(f"documents stress {add_d}")
    if add_w: bits.append(f"web adds {add_w}")
    bits.append(f"relevance → docs {rd}, web {rw}")
    return "; ".join(bits) + "."


def smart_final_answer(question: str, doc_p: str, web_p: str, judge: str) -> str:
    pool = []
    for chunk in [doc_p, web_p, judge]:
        if chunk and chunk != "—":
            pool.extend(sent_split(chunk))
    if not pool:
        return "—"
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    try:
        X = vec.fit_transform(pool + [question])
        sims = cosine_similarity(X[-1], X[:-1]).ravel()
    except Exception:
        sims = np.ones(len(pool))
    idxs = list(np.argsort(-sims))
    chosen, used = [], set()
    for i in idxs:
        s = gentle_paraphrase(pool[i])
        s_norm = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
        if len(s.split()) < 6:
            continue
        if s_norm in used:
            continue
        used.add(s_norm)
        chosen.append(s)
        if len(chosen) >= 4:
            break
    if not chosen:
        return "—"
    para = soft_wrap(chosen)
    words = para.split()
    if len(words) > 120:
        para = " ".join(words[:120]).rstrip(",;:") + "."
    if ("web relevance low" in judge.lower() or "docs relevance low" in judge.lower()) and len(chosen) >= 2:
        para += " Based on limited direct matches, this answer synthesizes the most reliable overlapping points."
    return para

# -------------------- Memory --------------------

def memory_load():
    try:
        if os.path.isfile(MEMORY_PATH):
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"qa": []}

def memory_save(mem):
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def memory_add(q: str, doc_p: str, web_p: str):
    mem = memory_load()
    qa = mem.get("qa", [])
    qa.append({"q": q, "doc": doc_p, "web": web_p, "ts": datetime.now().isoformat(timespec="seconds")})
    if len(qa) > MEMORY_LIMIT:
        qa = qa[-MEMORY_LIMIT:]
    mem["qa"] = qa
    memory_save(mem)

def memory_expand_query(q: str) -> str:
    mem = memory_load().get("qa", [])
    if not mem: return q
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    texts = [m["q"] for m in mem] + [q]
    try:
        X = vec.fit_transform(texts)
        sims = cosine_similarity(X[-1], X[:-1]).ravel()
        idx = int(np.argmax(sims)) if sims.size else -1
        if idx >= 0 and sims[idx] > 0.45:
            terms = key_terms(mem[idx].get("web") or mem[idx].get("doc") or "", 3)
            if terms:
                q2 = q + " " + " ".join(terms)
                return q2
    except Exception:
        return q
    return q

# -------------------- Engine call --------------------

def answer_once(q_raw: str):
    q_norm = normalize_query(q_raw)
    q_expanded = memory_expand_query(q_norm)
    _ = detect_intent(q_norm)

    doc_paragraph, doc_cites = doc_answer_paragraph(q_expanded)
    web_paragraph, web_urls = web_answer_paragraph(q_norm)
    jl = judge_line(q_norm, doc_paragraph, web_paragraph)
    sfa = smart_final_answer(q_norm, doc_paragraph, web_paragraph, jl)

    memory_add(q_norm, doc_paragraph, web_paragraph)
    return doc_paragraph, doc_cites, web_paragraph, web_urls, jl, sfa

# -------------------- UI --------------------
class RAGBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAGBot v9 — Tk GUI (Documents + Internet)")
        try:
            self.call("tk", "scaling", 1.6)
        except Exception:
            pass
        self.geometry("1500x900")
        self.minsize(1280, 820)
        self.configure(bg=DARK_BG)

        self.last_doc_cites = []
        self.last_web_urls = []

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        # Sidebar fixed width
        self.sidebar = ttk.Frame(root)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.sidebar.configure(width=440)
        self._style_theme()
        self._build_sidebar()

        # Main area
        self.main = ttk.Frame(root)
        self.main.pack(side="right", fill="both", expand=True)
        self._build_main()

        # Index docs and start clocks
        threading.Thread(target=self._index_docs, daemon=True).start()
        self.after(200, self._tick_clocks)

    def _style_theme(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Sidebar.TFrame", background=DARK_BG)
        s.configure("Card.TFrame", background=CARD_BG)
        s.configure("Main.TFrame", background="#ffffff")
        s.configure("H1.TLabel", font=("Segoe UI", 20, "bold"), foreground=TEXT, background=CARD_BG)
        s.configure("H2.TLabel", font=("Segoe UI", 16, "bold"), foreground=TEXT, background=CARD_BG)
        s.configure("Body.TLabel", font=("Segoe UI", 11), foreground=TEXT, background=CARD_BG)
        s.configure("Mute.TLabel", font=("Segoe UI", 11), foreground=SUBTLE, background=CARD_BG)
        s.configure("Accent.TLabel", font=("Segoe UI", 12, "bold"), foreground=ACCENT, background=CARD_BG)

    # ----- Sidebar -----
    def _card(self, parent):
        f = ttk.Frame(parent, style="Card.TFrame")
        f.pack(fill="x", padx=18, pady=(14,10))
        f.columnconfigure(0, weight=1)
        return f

    def _build_sidebar(self):
        self.sidebar.configure(style="Sidebar.TFrame")

        # Header Card
        hdr = self._card(self.sidebar)
        # Avatar
        if PIL_OK and os.path.isfile(PROFILE_IMG):
            try:
                img = Image.open(PROFILE_IMG).resize((96,96), Image.LANCZOS)
                ph = ImageTk.PhotoImage(img)
                av = ttk.Label(hdr, image=ph, style="Card.TFrame")
                av.image = ph
                av.grid(row=0, column=0, sticky="w", pady=(4,6))
            except Exception:
                ttk.Label(hdr, text="[avatar]", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=(4,6))
        else:
            ttk.Label(hdr, text="[avatar]", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=(4,6))
        ttk.Label(hdr, text=APP_TITLE, style="H1.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(hdr, text=APP_SUBTITLE, style="Mute.TLabel").grid(row=2, column=0, sticky="w", pady=(2,0))

        # Country Cards Row
        row = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        row.pack(fill="x", padx=18, pady=(6,12))
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)

        self._country_card(row, INDIA_FLAG, "India", side="IN", col=0)
        self._country_card(row, UK_FLAG, "UK", side="UK", col=1)

        # Timezone button card
        tz = self._card(self.sidebar)
        btn = ttk.Button(tz, text="Timezone", command=self._open_timezone_dialog)
        btn.grid(row=0, column=0, sticky="ew")

    def _flag_img(self, path, size=(54,34)):
        if not PIL_OK:
            return None
        try:
            im = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
        except Exception:
            im = Image.new("RGB", size, "#1f2937")
        return ImageTk.PhotoImage(im)

    def _country_card(self, parent, flag_path, title, side, col):
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.grid(row=0, column=col, sticky="nsew", padx=(0,8) if col == 0 else (8,0))
        # header
        h = ttk.Frame(card, style="Card.TFrame")
        h.grid(row=0, column=0, sticky="w")
        img = self._flag_img(flag_path)
        if img:
            lbl = ttk.Label(h, image=img, style="Card.TFrame")
            lbl.image = img
            lbl.pack(side="left", padx=(0,8))
        ttk.Label(h, text=title, style="H2.TLabel").pack(side="left")
        # date/time/tz
        date_lbl = ttk.Label(card, text="—", style="Mute.TLabel")
        date_lbl.grid(row=1, column=0, sticky="w", pady=(6,0))
        time_lbl = ttk.Label(card, text="00:00:00", style="H1.TLabel")
        time_lbl.grid(row=2, column=0, sticky="w")
        tz_lbl = ttk.Label(card, text="TZ", style="Mute.TLabel")
        tz_lbl.grid(row=3, column=0, sticky="w", pady=(2,6))
        link = ttk.Label(card, text="Live News", style="Accent.TLabel", cursor="hand2")
        link.grid(row=4, column=0, sticky="w")
        if side == "IN":
            self.in_date, self.in_time, self.in_tz = date_lbl, time_lbl, tz_lbl
            link.bind("<Button-1>", lambda e: webbrowser.open_new("https://news.google.com/home?hl=en-IN&gl=IN&ceid=IN:en"))
        else:
            self.uk_date, self.uk_time, self.uk_tz = date_lbl, time_lbl, tz_lbl
            link.bind("<Button-1>", lambda e: webbrowser.open_new("https://news.google.com/home?hl=en-GB&gl=GB&ceid=GB:en"))

    # ----- Main area -----
    def _build_main(self):
        top = ttk.Frame(self.main, padding=8, style="Main.TFrame")
        top.pack(fill="x")
        self.docs_dir_var = tk.StringVar(value=DOCS_DIR)
        ttk.Label(top, text="Documents:", font=("Segoe UI", 10, "bold"), background="#ffffff").pack(side="left")
        ttk.Label(top, textvariable=self.docs_dir_var, background="#ffffff", foreground="#666").pack(side="left", padx=(6,16))
        ttk.Button(top, text="Reload Docs", command=self.reload_docs).pack(side="left")
        ttk.Button(top, text="Show Sources", command=self.show_sources).pack(side="left", padx=(8,0))

        qframe = ttk.Frame(self.main, padding=(8,0,8,8), style="Main.TFrame")
        qframe.pack(fill="x")
        self.q_var = tk.StringVar()
        self.q_entry = ttk.Entry(qframe, textvariable=self.q_var)
        self.q_entry.pack(side="left", fill="x", expand=True)
        self.q_entry.bind("<Return>", self.on_ask)
        ttk.Button(qframe, text="Ask", command=self.on_ask).pack(side="left", padx=(8,0))

        self.out = ScrolledText(self.main, wrap="word", font=("Consolas", 11))
        self.out.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self._append("RAGBot Tk ready.\n")
        self.out.config(state="disabled")

        self.status = tk.StringVar(value="Indexing documents…")
        ttk.Label(self.main, textvariable=self.status, anchor="w", style="Main.TFrame").pack(side="bottom", fill="x")

    # ----- Helpers -----
    def _append(self, text: str):
        self.out.config(state="normal")
        self.out.insert("end", text)
        self.out.see("end")
        self.out.config(state="disabled")

    def _index_docs(self):
        try:
            rows = load_all_documents(DOCS_DIR)
            DOC_INDEX.fit(rows)
            self.status.set(f"Indexed {len(rows)} chunks from: {DOCS_DIR}")
            self._append(f"Indexed {len(rows)} chunks from: {DOCS_DIR}\n")
        except Exception as e:
            self.status.set("Indexing failed.")
            messagebox.showerror("Indexing error", str(e))

    def reload_docs(self):
        self.status.set("Reloading documents…")
        threading.Thread(target=self._index_docs, daemon=True).start()

    def show_sources(self):
        win = tk.Toplevel(self)
        win.title("Sources")
        win.geometry("820x560")
        box = ScrolledText(win, wrap="word", font=("Consolas", 10))
        box.pack(fill="both", expand=True)
        box.insert("end", "Documents:\n")
        if self.last_doc_cites:
            for (src, pg) in self.last_doc_cites:
                box.insert("end", f"- {src}" + (f" (page {pg})" if pg is not None else "") + "\n")
        else:
            box.insert("end", "—\n")
        box.insert("end", "\nInternet:\n")
        if self.last_web_urls:
            for u in self.last_web_urls:
                box.insert("end", f"- {u}\n")
        else:
            box.insert("end", "—\n")
        box.config(state="disabled")

    # ----- Ask/Answer -----
    def on_ask(self, event=None):
        q = self.q_var.get().strip()
        if not q:
            return
        self.q_entry.selection_range(0, 'end')
        self.status.set("Thinking…")
        self._append(f"\n> {q}\n")
        threading.Thread(target=self._answer_thread, args=(q,), daemon=True).start()

    def _answer_thread(self, q_raw: str):
        try:
            doc_p, doc_c, web_p, web_u, jl, sfa = answer_once(q_raw)
            out = []
            out.append("\nFrom Documents: " + (doc_p or "—"))
            out.append("\nFrom Internet: " + (web_p or "—"))
            if jl:
                out.append("\nJudgement & Compare: " + jl)
            out.append("\nSmart Final Answer: " + (sfa or "—"))
            out.append("\n")
            self.last_doc_cites = doc_c
            self.last_web_urls = web_u
            self.after(0, lambda: self._append("".join(out)))
            self.after(0, lambda: self.status.set("Done."))
        except Exception as e:
            self.after(0, lambda: self._append("\nFrom Documents: —\nFrom Internet: —\nSmart Final Answer: —\n"))
            self.after(0, lambda: self.status.set("Error."))
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

    # ----- Sidebar time updates -----
    def _tick_clocks(self):
        now_utc = datetime.now(timezone.utc)
        if ZoneInfo:
            try:
                ist = now_utc.astimezone(ZoneInfo("Asia/Kolkata"))
                uk  = now_utc.astimezone(ZoneInfo("Europe/London"))
            except Exception:
                ist = uk = now_utc
        else:
            ist = uk = now_utc
        self.in_date.config(text=ist.strftime("%b %d, %Y"))
        self.in_time.config(text=ist.strftime("%H:%M:%S"))
        self.in_tz.config(text=ist.strftime("%Z") or "IST")
        self.uk_date.config(text=uk.strftime("%b %d, %Y"))
        self.uk_time.config(text=uk.strftime("%H:%M:%S"))
        self.uk_tz.config(text=uk.strftime("%Z") or "UK")
        self.after(1000, self._tick_clocks)

    def _open_timezone_dialog(self):
        now_utc = datetime.now(timezone.utc)
        if ZoneInfo:
            try:
                ist = now_utc.astimezone(ZoneInfo("Asia/Kolkata"))
                uk  = now_utc.astimezone(ZoneInfo("Europe/London"))
            except Exception:
                ist = uk = now_utc
        else:
            ist = uk = now_utc
        win = tk.Toplevel(self)
        win.title("Timezones")
        ttk.Label(win, text=f"India (IST): {ist.strftime('%a %d %b %Y %H:%M:%S %Z')}").pack(padx=12, pady=(10,4))
        ttk.Label(win, text=f"UK:          {uk.strftime('%a %d %b %Y %H:%M:%S %Z')}").pack(padx=12, pady=(0,10))

if __name__ == "__main__":
    app = RAGBotGUI()
    app.mainloop()
