# 📚 RAG 文件問答（Gradio · PWA · RWD）

上傳 PDF / DOCX，選擇檢索策略後直接提問。檢索使用 fastembed（ONNX）與多語言 MiniLM 嵌入模型，回答由 Groq 模型生成，可匯出 Word / CSV 問答紀錄。

## 檢索策略

混合搜尋（RRF）、語意搜尋、關鍵字（字元 n-gram TF-IDF）、重新排序、多查詢、壓縮、父子文檔、HyDE。

## 專案結構

```
├── app.py            # 入口（FastAPI + Gradio），含 manifest / Service Worker / 圖示路由
├── ui.py             # Gradio 介面、RWD 樣式、PWA <head>、Word / CSV 匯出
├── rag_engine.py     # 檢索與問答引擎
├── static/sw.js      # Service Worker
├── prefetch.py       # 建置時預先下載嵌入模型
├── requirements.txt
├── render.yaml       # Render Blueprint
└── .gitignore
```

## 本機執行

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_xxx        # 選填；不設定則由使用者在介面輸入
uvicorn app:app --port 7860
```

開啟 http://localhost:7860 。`localhost` 視為安全來源，Service Worker 在本機也能測試。

## 上傳到 GitHub

```bash
git init
git add .
git commit -m "RAG PWA"
git branch -M main
git remote add origin https://github.com/<你的帳號>/<repo名稱>.git
git push -u origin main
```

> 不要把 API Key 寫進程式碼或 commit；`.gitignore` 已排除 `.env`。

## 部署到 Render

### 方式 A：Blueprint（推薦）

1. Render Dashboard → **New → Blueprint**，選擇此 GitHub repo。
2. Render 會讀取 `render.yaml`，在 `GROQ_API_KEY` 欄位填入你的 Key（可留空，讓使用者自行輸入）。
3. 按 **Apply**，等待建置完成（首次需安裝套件並下載模型，約數分鐘）。

### 方式 B：手動建立 Web Service

| 欄位 | 值 |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt && python prefetch.py` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips="*"` |
| Health Check Path | `/healthz` |

環境變數：

| 名稱 | 說明 |
|---|---|
| `PYTHON_VERSION` | `3.11.9` |
| `GROQ_API_KEY` | 選填，伺服器預設 Key |
| `FASTEMBED_CACHE_PATH` | `/opt/render/project/src/.fastembed_cache`（讓預載的模型留在執行環境） |
| `OMP_NUM_THREADS` | `1`，壓低記憶體 |
| `EMBED_MODEL` | 選填，預設 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（須為 fastembed 支援的模型） |

### 記憶體不足時（512MB 方案）

- 看日誌中的 `[core] ... RSS=…MB` 與 `[load] ... RSS=…MB`，可得知載入模型與處理文件後的實際用量。
- 仍然 OOM 時，把環境變數 `EMBED_MODEL` 設為 `none`：完全不載入嵌入模型，所有語意類策略改以關鍵字檢索，記憶體需求大幅下降。
- 或升級到 2GB 以上方案，保留語意檢索。
- 建置指令必須包含 `python prefetch.py`；手動建立服務時，啟動指令請加上 `--proxy-headers --forwarded-allow-ips="*"`。

### 注意事項

- **記憶體**：已改用 ONNX 的 fastembed 並移除 PyTorch / ChromaDB，目標可在 512MB 執行。文件很大（數百頁）或多人同時上傳時仍可能逼近上限，建議升級到 2GB 方案；實際規格以 Render 方案頁為準。
- **Key 安全**：設定 `GROQ_API_KEY` 後，所有訪客在未輸入自己的 Key 時都會使用你的額度。公開網址請自行評估，或不設定此變數。伺服器 Key 不會傳到瀏覽器。
- **資料不落地**：每位使用者的文件與向量只存在自己的 session 記憶體，離開後即釋放，不寫入磁碟。
- **冷啟動**：免費方案閒置後休眠，喚醒時需重新載入模型，第一次開啟會較慢。
- **掃描檔 PDF**：沒有文字層的 PDF 需先 OCR。

## PWA 安裝方式

需要 HTTPS（Render 預設提供）。

- **Android / Chrome、桌面 Chrome / Edge**：網址列的安裝圖示，或選單 → 「安裝應用程式」。
- **iPhone / iPad（Safari）**：分享 → 「加入主畫面」。

PWA 的範圍：

- Service Worker 只快取殼層資源（離線頁、圖示、manifest）。Gradio 的 API 與串流請求不會被攔截。
- 問答需要網路與後端，離線時會顯示「目前離線」頁面，不提供離線問答。
- 修改 `static/sw.js` 後，請調整檔內 `CACHE` 版本字串，讓使用者取得新版。

## RWD 重點

- 桌面：左側設定、右側對話（兩欄）。
- 平板 / 手機（< 1024px）：改為單欄堆疊。
- 手機（< 640px）：對話區高度改用 `dvh`、按鈕自動換行；輸入框字級 16px，避免 iOS 聚焦時自動放大。
- 支援瀏海與底部手勢列的安全區（`viewport-fit=cover` + `env(safe-area-inset-*)`）。
