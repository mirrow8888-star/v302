"""RAG 引擎：文件載入、八種檢索策略、問答生成（輕量版，可在 512MB 記憶體執行）。
- Core：ONNX 嵌入模型（fastembed，不依賴 PyTorch），全站共用、只載入一次
- RAG：每位使用者（Gradio session）各自一份，文件、向量、Key、模型互不干擾
"""
import os, re, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from groq import Groq
from pypdf import PdfReader
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 設定 ──
# 多語言 MiniLM（約 220MB，384 維）；記憶體充足時可改 EMBED_MODEL（須為 fastembed 支援的模型）
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 300, 60, 120, 24, 7000   # 此模型適合較短片段
ENV_KEY = os.getenv("GROQ_API_KEY", "").strip()   # 伺服器預設 Key（選填）

# 模型快取預設放專案目錄：建置階段下載後，執行階段可直接使用（不依賴環境變數）
CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH") or str(Path(__file__).parent / ".fastembed_cache")
NO_EMBED = EMBED_MODEL.strip().lower() == "none"   # EMBED_MODEL=none：不載入嵌入模型，語意類策略退回關鍵字檢索


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            return int(next(l for l in f if l.startswith("VmRSS")).split()[1]) // 1024
    except Exception: return -1


STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf); buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p; fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


class Core:
    """全站共用的嵌入模型（ONNX，單執行緒以壓低記憶體）。"""

    def __init__(self):
        self.lock = threading.Lock()   # 序列化嵌入運算，避免多人同時推高記憶體
        self.embedder = None if NO_EMBED else TextEmbedding(model_name=EMBED_MODEL, threads=1, cache_dir=CACHE_DIR)
        print(f"[core] embedder={'off' if NO_EMBED else EMBED_MODEL} · RSS={rss_mb()}MB", flush=True)

    def enc(self, texts, query=False):
        out = []
        with self.lock:
            for i in range(0, len(texts), 16):
                out.extend(self.embedder.embed(texts[i:i + 16], batch_size=8))
        m = np.asarray(out, dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


_core, _core_lock = None, threading.Lock()


def get_core():
    global _core
    with _core_lock:
        if _core is None: _core = Core()
        return _core


class RAG:
    def __init__(self):
        self.emb, self.cemb, self.cpar = None, None, None      # 片段 / 子片段向量與子→父對照
        self.client, self.model, self.using_env_key = None, MODELS[0], False
        self.sid, self.name = None, ""
        self.chunks, self.labels, self.n_child, self.tfidf, self.tm = [], [], 0, None, None
        self.set_key("")

    # ── Groq ──
    def set_key(self, key):
        key = (key or "").strip()
        self.using_env_key = not key and bool(ENV_KEY)
        key = key or ENV_KEY
        self.client = Groq(api_key=key) if key else None

    def list_models(self):
        if not self.client: return []
        try:
            ids = [m.id for m in self.client.models.list().data if getattr(m, "active", True) is not False]
        except Exception: return []
        return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))

    def llm(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client: return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try: r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower(): raise
                kw.pop("reasoning_effort", None); kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err: raise
            return None

    # ── 文件載入 ──
    @property
    def core(self): return get_core()

    def close(self):
        """session 結束時釋放此使用者的資料。"""
        self.sid, self.chunks, self.labels, self.tm, self.emb, self.cemb, self.cpar = None, [], [], None, None, None, None

    def load(self, path):
        t0, path = time.time(), Path(path)
        if path.suffix.lower() not in (".pdf", ".docx"): return "✗ 僅支援 PDF / DOCX"
        try:
            segs = self._pdf(path) if path.suffix.lower() == ".pdf" else self._docx(path)
            pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
            if not pairs: return "✗ 無可擷取文字（可能是掃描檔，需先 OCR）"
            self.close()
            self.sid, self.name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}", path.name
            self.chunks, self.labels = [p[0] for p in pairs], [p[1] for p in pairs]
            if self.core.embedder is not None:
                self.emb = self.core.enc(self.chunks)
                kids = [(t, p) for p, par in enumerate(self.chunks) for t in split_text(par, CHILD, CHILD_OVERLAP)]
                self.n_child = len(kids)
                self.cemb, self.cpar = self.core.enc([k[0] for k in kids]), [k[1] for k in kids]
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            self.tm = self.tfidf.fit_transform(self.chunks)
            print(f"[load] {self.name} · {len(self.chunks)} 片段 · RSS={rss_mb()}MB", flush=True)
            return f"✓ {self.name}\n{len(self.chunks)} 片段 · {time.time() - t0:.1f}s"
        except Exception as e:
            return f"✗ 載入失敗：{type(e).__name__}: {e}"

    def _pdf(self, path):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(str(path)).pages, 1) if (t := p.extract_text() or "").strip()]

    def _docx(self, path):
        doc, segs, head, buf, nt = Document(str(path)), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc); txt = para.text.strip()
                if not txt: continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush(); head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1; rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t): cells.append(t)
                    if cells: rows.append(" | ".join(cells))
                if rows: flush(); segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def vec(self, qs, k, child=False):
        if self.core.embedder is None:                          # 省記憶體模式：以關鍵字檢索代替向量
            return [self.kw(q, k) for q in qs]
        mat = self.cemb if child else self.emb
        if mat is None or not len(mat): return [[] for _ in qs]
        sims = self.core.enc(qs, True) @ mat.T                  # 已正規化，內積即餘弦相似度
        top = np.argsort(-sims, axis=1)[:, :max(1, min(k, mat.shape[0]))]
        return [[int(self.cpar[j]) if child else int(j) for j in row] for row in top]

    def kw(self, q, k):
        s = (self.tm @ self.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)): sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, q, strategy, k):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec([q], n * 2)[0], self.kw(q, n * 2)], n)
        if strategy == "semantic": idx = self.vec([q], k)[0]
        elif strategy == "tfidf": idx = self.kw(q, k)
        elif strategy == "parent_child": idx = list(dict.fromkeys(self.vec([q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = self.llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec([q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = self.llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec([q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12)); idx = cands[:k]
            if self.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {self.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = self.llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                             15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else: idx = hybrid(k)
        hits = [(i, self.chunks[i]) for i in idx]
        if strategy == "compress" and self.client and hits:
            def comp(h):
                r = self.llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex: hits = list(ex.map(comp, hits))
        return hits

    def answer(self, q, strategy, k, model):
        if not self.chunks: return "請先上傳 PDF 或 DOCX 文件。", ""
        self.model, t0 = model or MODELS[0], time.time()
        hits = self.retrieve(q, strategy, int(k)); t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{self.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX: break
            blocks.append(b); used += len(b)
        ctx = "\n\n".join(blocks)
        src = f"📄 {self.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s\n{'─' * 40}\n\n{ctx}"
        if not hits: return "找不到相關片段，請換個問法或策略。", src
        if not self.client: return "⚠ 尚未設定 Groq API Key（檢索結果請看下方片段）", src
        try:
            ans = self.llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                           f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                           1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        except Exception as e:
            return f"✗ 生成失敗：{type(e).__name__}: {e}", src
        return f"{ans or '（模型未回傳內容）'}\n\n<sub>{self.model} · 檢索 {t1 - t0:.1f}s · 生成 {time.time() - t1:.1f}s</sub>", src
