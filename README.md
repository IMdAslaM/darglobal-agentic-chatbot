# DarGlobal & Wasalt Real Estate AI Chatbot

An agentic, containerized Python AI chatbot and RAG application that provides real-time property intelligence, comparison, and advisory across **DarGlobal** (London Stock Exchange-listed luxury branded residences) and **Wasalt** (Saudi Arabia's premier REGA-licensed property marketplace).

---

## 🌟 Key Features

- **Multi-Portal Intelligence**:
  - **DarGlobal**: Ultra-luxury branded residences partnering with Pagani, Lamborghini, Missoni, Aston Martin, Elie Saab, and Trump across Dubai, London, Muscat, Doha, Marbella, and Riyadh.
  - **Wasalt**: Verified Saudi Arabian villas, penthouses, and apartments across prime districts in Riyadh (Al Malqa, Hittin, Al Narjis, Al Yasmin) and Jeddah (Al Shati, Obhur, Al Mohammedia).
- **RAG & Agentic Retrieval**:
  - Fast hybrid search combining exact keyword/brand boosts with BM25 semantic scoring.
  - Contextual real estate advisory prompt acting as an elite property consultant.
  - Direct citations of developer/portal, pricing in local currencies (AED, SAR, GBP, EUR) and USD approximations.
- **OpenRouter Multi-Model Support**:
  - Powered by the OpenAI Python SDK pointed to OpenRouter (`https://openrouter.ai/api/v1`).
  - Supports top free models: `meta-llama/llama-3.3-70b-instruct:free`, `google/gemma-2-9b-it:free`, `meta-llama/llama-3.1-8b-instruct:free`, `mistralai/mistral-7b-instruct:free`, and `qwen/qwen-2.5-72b-instruct:free`.
  - Built-in automatic model failover and offline fallback engine.
- **Modern Luxury UI**:
  - Dark/gold executive aesthetic inspired by international real estate portfolios.
  - Real-time conversation with quick suggestion prompt chips.
  - Dynamic "Matched Properties" panel rendering interactive cards with high-res photos, specs, and direct listing links.
  - Integrated property directory catalog modal.
- **Scraper Engine (`scraper.py`)**:
  - Scrapes listings from `darglobal.co.uk` and `wasalt.com` / `wasalt.sa` using `httpx` and `BeautifulSoup`.
  - Maintains and expands `data/properties.json`.
- **Production-Ready Containerization**:
  - Hardened Dockerfile with non-root security user, health checks, and pre-bundled dataset.
  - Default port `7860` for 1-click deployment on Hugging Face Spaces or Docker containers.

---

## 🏗️ Project Structure

```text
├── app.py                  # FastAPI server & REST API endpoints
├── rag.py                  # RAG retrieval & OpenRouter LLM engine
├── scraper.py              # Web scraper for DarGlobal & Wasalt
├── requirements.txt        # Python package dependencies
├── Dockerfile              # Production Docker container definition
├── .env.example            # Environment variables template
├── .env                    # Local secrets (ignored by Git)
├── .gitignore              # Git ignore rules
├── data/
│   └── properties.json     # Curated & scraped property database
└── static/
    └── index.html          # Responsive single-page luxury Web UI
```

---

## 🚀 Quickstart Guide

### 1. Workspace & Git Configuration

Clone the repository and set your local Git credentials (isolated to this repository):

```bash
git clone https://github.com/IMdAslaM/darglobal-agentic-chatbot.git
cd darglobal-agentic-chatbot

# Prevent credential leaks: configure Git locally without --global
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

### 2. Environment Variables Setup

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key (get one for free at [openrouter.ai/keys](https://openrouter.ai/keys)):

```env
OPENROUTER_API_KEY=sk-or-v1-your-actual-openrouter-key-here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
PORT=7860
HOST=0.0.0.0
```

### 3. Install Dependencies

Create and activate a virtual environment:

```bash
# macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 4. Run the Scraper

Run `scraper.py` once to refresh or verify the scraped property dataset in `data/properties.json`:

```bash
python scraper.py
```

Options:
- `--source all`: Scrapes both DarGlobal and Wasalt.
- `--source darglobal`: Scrapes only DarGlobal projects.
- `--source wasalt`: Scrapes only Wasalt listings.
- `--output data/properties.json`: Custom output destination.

### 5. Launch the Chatbot Locally

Start the FastAPI application:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Open your browser and navigate to:
```
http://localhost:7860
```

---

## 🐳 Docker Containerization

### Build Docker Image

```bash
docker build -t darglobal-wasalt-chatbot .
```

### Run Docker Container

```bash
docker run -p 7860:7860 --env-file .env darglobal-wasalt-chatbot
```

Access the app at `http://localhost:7860`.

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web Chat UI |
| `POST` | `/api/chat` | RAG query endpoint with context retrieval |
| `GET` | `/api/properties` | List properties with portal, city, and search filters |
| `GET` | `/api/properties/{id}` | Get detailed specs for a specific property |
| `POST` | `/api/scrape` | Trigger background scraper update |
| `GET` | `/api/health` | Health check & model status |

---

## 📦 Push to GitHub

When ready to push your project to GitHub:

```bash
git add .
git commit -m "Build DarGlobal and Wasalt agentic chatbot"
git push origin main
```

*(When prompted for GitHub authentication, use your GitHub Personal Access Token instead of your account password).*

---

## 📄 License
MIT License.
