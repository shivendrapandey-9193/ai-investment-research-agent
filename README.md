# 🚀 AI Investment Research Agent

> **AI-Powered Multi-Agent Investment Research Platform** that analyzes stocks using real-time financial data, market news, sentiment analysis, SWOT analysis, risk assessment, and AI-driven investment recommendations.

---

## 📖 Overview

**AI Investment Research Agent** is an intelligent financial analysis platform built using **FastAPI, LangGraph, LangChain, Google Gemini, Groq, Yahoo Finance, NewsAPI, and Serper API**.

The application uses multiple AI agents working together to analyze companies from different perspectives and generate comprehensive investment recommendations.

The system combines:

* 📊 Financial Analysis
* 📰 Market News Intelligence
* 😊 Sentiment Analysis
* 🔍 SWOT Analysis
* ⚠️ Risk Assessment
* 🤖 Multi-Agent Reasoning
* 💡 AI-Powered Investment Recommendations

---

# ✨ Features

## 📊 Financial Analysis Agent

* Revenue Growth Analysis
* Profit Margin Evaluation
* Earnings Performance
* Market Capitalization Analysis
* Price-to-Earnings (P/E) Ratio
* 52-Week High / Low Analysis
* Company Fundamentals
* Sector & Industry Insights

---

## 📰 News Intelligence Agent

* Real-Time Financial News Collection
* Company-Specific News
* AI-Based News Summarization
* Positive & Negative Catalyst Detection
* News Impact Analysis
* Market Sentiment Evaluation

---

## 😊 Sentiment Analysis Agent

* News Sentiment Classification
* Positive Sentiment Detection
* Negative Sentiment Detection
* Neutral Sentiment Detection
* Investment Sentiment Score

---

## 🔍 SWOT Analysis Agent

Automatically generates:

* ✅ Strengths
* ❌ Weaknesses
* 🚀 Opportunities
* ⚠️ Threats

---

## ⚠️ Risk Assessment Agent

Evaluates

* Market Risk
* Valuation Risk
* Regulatory Risk
* Competitive Risk
* Business Risk
* Execution Risk

---

## 🏛 Investment Committee Agent

The platform uses multiple AI agents that collaborate before producing the final recommendation.

| Agent           | Responsibility             |
| --------------- | -------------------------- |
| Financial Agent | Fundamental Analysis       |
| News Agent      | Market News & Sentiment    |
| SWOT Agent      | Business Strategy Analysis |
| Risk Agent      | Risk Evaluation            |
| Decision Agent  | Final Recommendation       |

Final Recommendations:

🟢 **INVEST**

🟡 **WATCH**

🔴 **PASS**

---

# 🏗 System Architecture

```text
                    User
                     │
                     ▼
           Research Request
                     │
                     ▼
          Multi-Agent Workflow
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
Financial Agent   News Agent   Sentiment Agent
      │              │
      └──────┬───────┘
             ▼
        SWOT Agent
             ▼
        Risk Agent
             ▼
      Decision Agent
             ▼
 Investment Recommendation
```

---

# 📈 Recommendation Scale

| Score    | Recommendation |
| -------- | -------------- |
| 70 - 100 | 🟢 INVEST      |
| 40 - 69  | 🟡 WATCH       |
| 0 - 39   | 🔴 PASS        |

---

# 🛠 Tech Stack

## Backend

* FastAPI
* Uvicorn

## AI Frameworks

* LangChain
* LangGraph

## LLM Providers

* Google Gemini 1.5 Flash
* Groq Llama 3.3

## Financial Data

* Yahoo Finance (yFinance)

## News Sources

* NewsAPI
* Serper API

## Python Libraries

* Requests
* Pandas
* NumPy
* python-dotenv
* Pydantic

---

# 📂 Project Structure

```text
AI-Investment-Research-Agent/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
├── agents/
│   ├── financial_agent.py
│   ├── news_agent.py
│   ├── sentiment_agent.py
│   ├── swot_agent.py
│   ├── risk_agent.py
│   └── decision_agent.py
│
├── utils/
│   ├── yahoo_finance.py
│   ├── news.py
│   └── helper.py
│
└── workflow/
    └── langgraph_workflow.py
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/AI-Investment-Research-Agent.git

cd AI-Investment-Research-Agent
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
GEMINI_API_KEY=your_gemini_api_key

GROQ_API_KEY=your_groq_api_key

NEWS_API_KEY=your_news_api_key

SERPER_API_KEY=your_serper_api_key
```

---

# ▶️ Run the Application

```bash
uvicorn app:app --reload
```

Application URL

```
http://127.0.0.1:8000
```

---

# 📡 API Endpoints

## Health Check

```
GET /health
```

---

## Analyze Company

```
POST /analyze
```

Request

```json
{
    "company":"Tesla"
}
```

Sample Response

```json
{
    "company":"Tesla",
    "recommendation":"WATCH",
    "score":67,
    "financial_analysis":{},
    "news_analysis":{},
    "risk_analysis":{},
    "swot_analysis":{}
}
```

---

# 📚 Data Sources

This project collects financial information from publicly available APIs and trusted services.

### 📊 Yahoo Finance (yFinance)

Used for

* Stock Prices
* Company Financial Statements
* Market Capitalization
* P/E Ratio
* Revenue
* Earnings
* Company Fundamentals
* Historical Stock Data

---

### 📰 NewsAPI

Used for

* Company News
* Financial News
* Market Headlines

---

### 🔍 Serper API

Used for

* Google Search Results
* Latest Company News
* Additional Market Intelligence

---

### 🤖 AI Models

Google Gemini

* Investment Analysis
* SWOT Generation
* Final Recommendation
* Company Evaluation

Groq (Llama 3.3)

* Fast AI Inference
* Financial Reasoning
* Agent Collaboration

---

# ⚡ Workflow

```text
User Company Name
        │
        ▼
Financial Data Collection
        │
        ▼
Market News Collection
        │
        ▼
Sentiment Analysis
        │
        ▼
SWOT Analysis
        │
        ▼
Risk Assessment
        │
        ▼
AI Decision Agent
        │
        ▼
Final Recommendation
```

---

# 🎯 Key Achievements

* ✅ Built a Multi-Agent AI Investment Analysis System
* ✅ Integrated Real-Time Financial Data
* ✅ Implemented News Sentiment Analysis
* ✅ Automated SWOT Generation
* ✅ Automated Risk Assessment
* ✅ AI-Based Investment Recommendation Engine
* ✅ Built REST APIs using FastAPI
* ✅ LangGraph Multi-Agent Workflow
* ✅ Real-Time Company Research

---

# 🚀 Future Enhancements

* Portfolio Management
* Technical Analysis Agent
* Earnings Call Analysis
* SEC Filing Analysis
* Historical Backtesting
* Interactive Dashboard
* Mobile Application
* Portfolio Risk Optimization
* Cryptocurrency Analysis
* Email Report Generation

---

# 📌 Disclaimer

This project is developed for **educational and research purposes only**.

The investment recommendations generated by this system are AI-assisted insights based on publicly available financial information and should **not** be considered professional financial advice.

Always perform your own research before making investment decisions.

---

# 👨‍💻 Author

**Shivendra Pandey**

Computer Science Student

AI & Machine Learning Enthusiast

---

# ⭐ Support

If you found this project helpful,

⭐ Star the repository

🍴 Fork the repository

🛠 Contribute to the project

📢 Share it with others

---

# 📄 License

This project is licensed under the **MIT License**.

---

# ❤️ Acknowledgements

Special thanks to the following open-source tools and APIs that made this project possible:

* FastAPI
* LangChain
* LangGraph
* Google Gemini
* Groq
* Yahoo Finance (yFinance)
* NewsAPI
* Serper API
* Python Open Source Community

---

## 🚀 Empowering Smarter Investment Decisions with AI
