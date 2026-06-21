# 🚀 AI Investment Research Agent

<div align="center">

# AI-Powered Multi-Agent Investment Research Platform

Analyze stocks using real-time financial data, market news, sentiment analysis, SWOT analysis, risk assessment, and AI-driven investment recommendations.

</div>

---

## 📖 Overview

AI Investment Research Agent is an intelligent financial analysis system built using FastAPI, LangGraph, LangChain, Google Gemini, Groq, Yahoo Finance, and real-time news sources.

The platform combines multiple specialized AI agents to evaluate companies from different perspectives and generate actionable investment recommendations.

### Key Capabilities

- Real-time Financial Analysis
- Market News Intelligence
- Sentiment Analysis
- SWOT Analysis
- Risk Assessment
- Multi-Agent Decision Making
- AI-Powered Investment Recommendations

---

## ✨ Features

### 📊 Financial Analysis Agent

- Revenue Growth Analysis
- Profit Margin Evaluation
- Market Capitalization Analysis
- P/E Ratio Assessment
- 52-Week High/Low Tracking
- Sector & Industry Analysis

### 📰 News Intelligence Agent

- Real-Time News Collection
- Financial News Filtering
- Sentiment Analysis
- Positive & Negative Catalyst Detection
- News Impact Scoring

### ⚠️ Risk Assessment Agent

- Regulatory Risk Analysis
- Competitive Risk Evaluation
- Valuation Risk Assessment
- Market Risk Analysis
- Execution Risk Detection

### 🔍 SWOT Analysis Agent

- Strengths
- Weaknesses
- Opportunities
- Threats

### 🏛️ Investment Committee Agent

| Agent | Responsibility |
|---------|---------------|
| Financial Agent | Fundamental Analysis |
| News Agent | Sentiment Evaluation |
| Risk Agent | Risk Assessment |
| SWOT Agent | Strategic Analysis |

Produces a final recommendation:

- 🟢 INVEST
- 🟡 WATCH
- 🔴 PASS

---

## 🏗 Architecture

```text
User Request
      │
      ▼
Research Agent
      │
 ┌────┼────┐
 ▼         ▼

Financial  News
Agent      Agent

      ▼
 SWOT Agent
      ▼
 Risk Agent
      ▼
Decision Agent
      ▼
Recommendation
```

---

## 🛠 Tech Stack

### Backend
- FastAPI
- Uvicorn

### AI Frameworks
- LangChain
- LangGraph

### LLM Providers
- Google Gemini 1.5 Flash
- Groq Llama 3.3

### Financial Data
- Yahoo Finance (yFinance)

### News Sources
- NewsAPI
- Serper API

---

## 📂 Project Structure

```text
AI-Investment-Research-Agent/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
└── Multi-Agent Workflow
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/ai-investment-research-agent.git
cd ai-investment-research-agent
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
NEWS_API_KEY=your_news_api_key
```

---

## ▶️ Run Application

```bash
uvicorn app:app --reload
```

Application URL:

```text
http://127.0.0.1:8000
```

---

## 📡 API Endpoints

### Health Check

```http
GET /health
```

### Analyze Company

```http
POST /analyze
```

Request:

```json
{
  "company": "Tesla"
}
```

---

## 📈 Recommendation Scale

| Score | Recommendation |
|---------|---------------|
| 70 - 100 | 🟢 INVEST |
| 40 - 69 | 🟡 WATCH |
| 0 - 39 | 🔴 PASS |

---

## 🎯 Key Achievements

- Built a Multi-Agent AI Investment Analysis System
- Integrated Real-Time Financial Data
- Implemented News Sentiment Analysis
- Developed Deterministic Scoring Engine
- Created REST APIs with FastAPI
- Automated SWOT and Risk Assessment

---

## 🚀 Future Enhancements

- Portfolio Management
- Technical Analysis Agent
- Earnings Call Analysis
- SEC Filing Analysis
- Historical Backtesting
- Interactive Dashboard
- Mobile Application

---

## 👨‍💻 Author

**Shivendra Pandey**

Computer Science Student | AI & Machine Learning Enthusiast

---

## ⭐ Support

If you found this project useful:

- Star this repository
- Fork this repository
- Share with others

---

## 📜 License

Licensed under the MIT License.

---

### Empowering Smarter Investment Decisions with AI 🚀
