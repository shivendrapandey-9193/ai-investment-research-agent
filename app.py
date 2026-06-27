import os
import json
import re
import logging
import asyncio
from typing import Dict, List, Optional, TypedDict

import httpx
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("investment_agent")

# ========================= API KEYS (set these in your local .env) =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

FINANCE_DOMAINS = [
    "reuters.com", "finance.yahoo.com", "marketwatch.com", "cnbc.com",
    "investing.com", "bloomberg.com", "fool.com", "wsj.com", "ft.com",
    "barrons.com", "seekingalpha.com", "morningstar.com", "benzinga.com",
    "businesswire.com", "prnewswire.com", "economictimes.indiatimes.com",
    "moneycontrol.com", "livemint.com", "business-standard.com",
    "apnews.com", "forbes.com", "zacks.com", "nasdaq.com", "finviz.com",
]


def is_finance_domain(url: str) -> bool:
    return any(domain in url.lower() for domain in FINANCE_DOMAINS)


logger.info(f"GEMINI: {'OK' if GEMINI_API_KEY else 'MISSING'}")
logger.info(f"GROQ: {'OK' if GROQ_API_KEY else 'MISSING'}")
logger.info(f"SERPER: {'OK' if SERPER_API_KEY else 'MISSING'}")
logger.info(f"NEWS_API: {'OK' if NEWS_API_KEY else 'MISSING'}")
logger.info(f"ALPHA_VANTAGE: {'OK' if ALPHA_VANTAGE_API_KEY else 'MISSING'}")
logger.info(f"FINNHUB: {'OK' if FINNHUB_API_KEY else 'MISSING'}")
logger.info(f"FMP: {'OK' if FMP_API_KEY else 'MISSING'}")


# ========================= MODELS =========================
class AnalyzeRequest(BaseModel):
    company: str = Field(..., min_length=1, description="Company name", examples=["Tesla"])

    @field_validator("company")
    @classmethod
    def strip_company(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("company cannot be empty")
        return v


class SWOT(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)


class NewsAnalysis(BaseModel):
    sentiment: str = "Neutral"
    score: int = 50
    positives: List[str] = Field(default_factory=list)
    negatives: List[str] = Field(default_factory=list)
    key_highlights: List[str] = Field(default_factory=list)


class FinancialSnapshot(BaseModel):
    symbol: Optional[str] = None
    summary: str = ""
    publicly_traded: bool = True
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    current_price: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    industry: Optional[str] = None
    sector: Optional[str] = None
    headquarters: Optional[str] = None
    ceo: Optional[str] = None
    website: Optional[str] = None
    company_name: Optional[str] = None


class RiskAssessment(BaseModel):
    risk_score: int = 50
    risks: List[str] = Field(default_factory=list)


class CommitteeVote(BaseModel):
    financial_agent: str = "WATCH"
    news_agent: str = "WATCH"
    risk_agent: str = "WATCH"
    swot_agent: str = "WATCH"
    final_decision: str = "WATCH"


class AnalysisResponse(BaseModel):
    company: str
    recommendation: str = "PASS"
    overall_score: int = Field(..., ge=0, le=100)
    confidence_score: int = Field(..., ge=0, le=100)
    confidence_reason: str = ""
    news_sentiment: str = "Neutral"
    executive_summary: str = ""
    financial_analysis: str = ""
    financial_data: FinancialSnapshot = Field(default_factory=FinancialSnapshot)
    risk_assessment: str = ""
    swot: SWOT = Field(default_factory=SWOT)
    news_analysis: NewsAnalysis = Field(default_factory=NewsAnalysis)
    risk: RiskAssessment = Field(default_factory=RiskAssessment)
    investment_committee: CommitteeVote = Field(default_factory=CommitteeVote)
    bull_case: str = ""
    bear_case: str = ""
    reasoning: str = ""
    sources: List[str] = Field(default_factory=list)


# ========================= LLM SERVICE =========================
class LLMService:
    """
    Wraps Gemini (primary) and Groq (fallback). Lazily imports each
    provider's SDK so a missing package / key never crashes the app.
    """

    def __init__(self):
        self.gemini = None
        self.groq = None

        if GEMINI_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.gemini = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.2,
                )
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        if GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                self.groq = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=GROQ_API_KEY,
                    temperature=0.2,
                )
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")

    @staticmethod
    def _extract_json(content: str) -> Dict:
        content = content.strip()
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]
        content = content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        return json.loads(content)

    async def _call(self, model, prompt: str, system: Optional[str]) -> Dict:
        msgs = []
        if system:
            msgs.append(SystemMessage(content=system))
        msgs.append(HumanMessage(content=prompt))
        resp = await model.ainvoke(msgs)
        return self._extract_json(resp.content)

    async def generate(self, prompt: str, system: Optional[str] = None) -> Dict:
        if self.gemini:
            try:
                return await self._call(self.gemini, prompt, system)
            except Exception as e:
                logger.warning(f"Gemini call failed, trying Groq: {e}")
        if self.groq:
            try:
                return await self._call(self.groq, prompt, system)
            except Exception as e:
                logger.warning(f"Groq call failed: {e}")
        logger.error("No LLM provider available or all calls failed — returning empty dict")
        return {}


llm = LLMService()


# ========================= RESEARCH (REAL-TIME) =========================
async def serper_search(client: httpx.AsyncClient, query: str, num: int = 10) -> Dict:
    if not SERPER_API_KEY:
        return {}
    try:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num},
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Serper search error: {e}")
        return {}


async def newsapi_search(client: httpx.AsyncClient, company: str) -> List[Dict]:
    if not NEWS_API_KEY:
        return []
    try:
        resp = await client.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": company,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 10,
                "apiKey": NEWS_API_KEY,
            },
        )
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception as e:
        logger.error(f"NewsAPI error: {e}")
        return []


async def research_agent(company: str) -> Dict:
    sources: List[str] = []
    news_text = ""
    company_lower = company.lower().strip()

    def is_relevant(text: str) -> bool:
        if not text:
            return False
        return company_lower in text.lower()

    async def run_pass(serper_query: str) -> tuple[List[str], str]:
        pass_sources: List[str] = []
        pass_text = ""
        async with httpx.AsyncClient(timeout=15.0) as client:
            serper_data, news_articles = await asyncio.gather(
                serper_search(client, serper_query),
                newsapi_search(client, company),
            )

        for item in serper_data.get("organic", [])[:10]:
            link = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if not link or not is_finance_domain(link):
                continue
            if not is_relevant(title + " " + snippet):
                continue
            pass_sources.append(link)
            if snippet:
                pass_text += snippet + "\n"

        for article in news_articles[:10]:
            url = article.get("url", "")
            title = article.get("title") or ""
            desc = article.get("description") or ""
            if not url or not is_finance_domain(url):
                continue
            if not is_relevant(title + " " + desc):
                continue
            pass_sources.append(url)
            if title or desc:
                pass_text += f"{title}. {desc}\n"

        return pass_sources, pass_text

    sources, news_text = await run_pass(f'"{company}" stock financial news earnings analysis')

    if not news_text.strip():
        sources, news_text = await run_pass(f'"{company}" latest company news')

    seen = set()
    deduped_sources = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            deduped_sources.append(s)

    has_real_news = bool(news_text.strip()) and not news_text.startswith("No live news")

    if not news_text.strip():
        news_text = (
            f"No live news could be retrieved for {company}. "
            "This may be due to missing SERPER_API_KEY / NEWS_API_KEY or no current coverage."
        )

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100, separator="\n")
    chunks = splitter.split_text(news_text)

    return {"sources": deduped_sources, "news_chunks": chunks, "raw_news": news_text, "has_real_news": has_real_news}


# ========================= FINANCIAL DATA (REAL-TIME, multi-provider) =========================
TICKER_MAP = {
    "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS", "icici bank": "ICICIBANK.NS",
}

# Known well-funded private companies that yfinance/other providers will
# otherwise mis-match to an unrelated ticker (e.g. "spacex" -> SPCX ETF).
PRIVATE_COMPANIES = {
    "spacex", "openai", "stripe", "byju's", "byjus", "byju", "anthropic",
    "bytedance", "epic games", "databricks", "canva", "shein", "discord",
    "automattic", "instacart's parent", "revolut", "klarna", "xai",
}


def is_private_company(company: str) -> bool:
    return company.lower().strip() in PRIVATE_COMPANIES


def resolve_symbol(company: str) -> tuple[str, bool]:
    """Returns (symbol, is_trusted). is_trusted=True means the symbol came
    from our curated map and should bypass the name-matching guard below —
    that guard previously broke on cases like Google->GOOGL, whose yfinance
    longName is "Alphabet Inc." and therefore failed a naive name check."""
    key = company.lower().strip()
    if key in TICKER_MAP:
        return TICKER_MAP[key], True
    return company.upper().replace(" ", ""), False


def _name_matches(company_name: str, info: Dict) -> bool:
    """Guard against an unrelated ticker/ETF match for untrusted/typed symbols."""
    long_name = (info.get("longName") or info.get("shortName") or "").lower()
    if not long_name:
        return True  # can't verify, don't block
    first_word = company_name.lower().split()[0]
    return first_word in long_name or long_name.split()[0] in company_name.lower()


def _extract_ceo_yf(info: Dict) -> Optional[str]:
    """yfinance exposes officers as a list of dicts; CEO title varies by company."""
    officers = info.get("companyOfficers") or []
    for officer in officers:
        title = (officer.get("title") or "").lower()
        if "chief executive officer" in title or title.strip() == "ceo":
            return officer.get("name")
    # Fallback: first officer if no exact CEO title match
    if officers:
        return officers[0].get("name")
    return None


def _yfinance_fetch(company: str, symbol: str, trusted: bool):
    """Runs in a thread (yfinance is sync/blocking)."""
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    has_price = info.get("regularMarketPrice") is not None or info.get("currentPrice") is not None

    if info and has_price and (trusted or _name_matches(company, info)):
        info["_ceo"] = _extract_ceo_yf(info)
        return symbol, info

    if not trusted:
        try:
            search = yf.Search(company, max_results=3)
            for q in search.quotes:
                alt_symbol = q.get("symbol")
                if not alt_symbol:
                    continue
                info2 = yf.Ticker(alt_symbol).info or {}
                if info2 and _name_matches(company, info2):
                    info2["_ceo"] = _extract_ceo_yf(info2)
                    return alt_symbol, info2
        except Exception as e:
            logger.warning(f"yfinance search fallback failed: {e}")

    return symbol, {}


async def _alpha_vantage_fetch(client: httpx.AsyncClient, symbol: str) -> Optional[Dict]:
    if not ALPHA_VANTAGE_API_KEY:
        return None
    try:
        resp = await client.get(
            "https://www.alphavantage.co/query",
            params={"function": "OVERVIEW", "symbol": symbol, "apikey": ALPHA_VANTAGE_API_KEY},
        )
        data = resp.json()
        if not data or "Symbol" not in data:
            return None

        def _f(key):
            try:
                val = data.get(key)
                return float(val) if val and val != "None" else None
            except (ValueError, TypeError):
                return None

        return {
            "longName": data.get("Name"),
            "currentPrice": _f("AnalystTargetPrice"),
            "marketCap": _f("MarketCapitalization"),
            "trailingPE": _f("PERatio"),
            "fiftyTwoWeekHigh": _f("52WeekHigh"),
            "fiftyTwoWeekLow": _f("52WeekLow"),
            "revenueGrowth": _f("QuarterlyRevenueGrowthYOY"),
            "profitMargins": _f("ProfitMargin"),
            "industry": data.get("Industry"),
            "sector": data.get("Sector"),
            "city": None, "state": None, "country": data.get("Country"),
        }
    except Exception as e:
        logger.warning(f"Alpha Vantage fetch failed: {e}")
        return None


async def _finnhub_fetch(client: httpx.AsyncClient, symbol: str) -> Optional[Dict]:
    if not FINNHUB_API_KEY:
        return None
    try:
        profile_resp, quote_resp, metrics_resp = await asyncio.gather(
            client.get("https://finnhub.io/api/v1/stock/profile2",
                       params={"symbol": symbol, "token": FINNHUB_API_KEY}),
            client.get("https://finnhub.io/api/v1/quote",
                       params={"symbol": symbol, "token": FINNHUB_API_KEY}),
            client.get("https://finnhub.io/api/v1/stock/metric",
                       params={"symbol": symbol, "metric": "all", "token": FINNHUB_API_KEY}),
        )
        profile = profile_resp.json() or {}
        quote = quote_resp.json() or {}
        metrics = (metrics_resp.json() or {}).get("metric", {}) or {}

        if not profile or not profile.get("name"):
            return None

        market_cap = profile.get("marketCapitalization")
        return {
            "longName": profile.get("name"),
            "currentPrice": quote.get("c"),
            "marketCap": (market_cap * 1_000_000) if market_cap else None,
            "trailingPE": metrics.get("peExclExtraTTM"),
            "fiftyTwoWeekHigh": metrics.get("52WeekHigh"),
            "fiftyTwoWeekLow": metrics.get("52WeekLow"),
            "revenueGrowth": metrics.get("revenueGrowthTTMYoy"),
            "profitMargins": metrics.get("netProfitMarginTTM"),
            "industry": profile.get("finnhubIndustry"),
            "sector": None,
            "city": None, "state": None, "country": profile.get("country"),
            "_ceo": None,  # Finnhub free tier profile2 doesn't include CEO name
            "website": profile.get("weburl"),
        }
    except Exception as e:
        logger.warning(f"Finnhub fetch failed: {e}")
        return None


async def _fmp_fetch(client: httpx.AsyncClient, symbol: str) -> Optional[Dict]:
    if not FMP_API_KEY:
        return None
    try:
        resp = await client.get(
            f"https://financialmodelingprep.com/api/v3/profile/{symbol}",
            params={"apikey": FMP_API_KEY},
        )
        data = resp.json()
        if not data or not isinstance(data, list) or not data[0].get("companyName"):
            return None
        d = data[0]
        return {
            "longName": d.get("companyName"),
            "currentPrice": d.get("price"),
            "marketCap": d.get("mktCap"),
            "trailingPE": None,
            "fiftyTwoWeekHigh": None,
            "fiftyTwoWeekLow": None,
            "revenueGrowth": None,
            "profitMargins": None,
            "industry": d.get("industry"),
            "sector": d.get("sector"),
            "city": d.get("city"), "state": d.get("state"), "country": d.get("country"),
            "_ceo": d.get("ceo"),
            "website": d.get("website"),
        }
    except Exception as e:
        logger.warning(f"FMP fetch failed: {e}")
        return None


async def financial_agent(company: str) -> Dict:
    if is_private_company(company):
        return {
            "symbol": None,
            "publicly_traded": False,
            "summary": f"{company} is a privately held company — no public market data available.",
            "industry": None, "sector": None, "headquarters": None,
        }

    symbol, trusted = resolve_symbol(company)

    try:
        resolved_symbol, info = await asyncio.to_thread(_yfinance_fetch, company, symbol, trusted)

        # Fallback chain: Alpha Vantage -> Finnhub -> FMP, only if yfinance
        # came back empty (rate-limited, mismatched, transient error, etc.)
        if not info:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for fetcher in (_alpha_vantage_fetch, _finnhub_fetch, _fmp_fetch):
                    info = await fetcher(client, symbol)
                    if info:
                        resolved_symbol = symbol
                        logger.info(f"Used fallback provider {fetcher.__name__} for {company}")
                        break

        if not info:
            return {
                "symbol": None,
                "publicly_traded": False,
                "summary": f"Could not retrieve market data for '{company}' from any provider — "
                           f"treating as unavailable rather than guessing.",
                "industry": None, "sector": None, "headquarters": None,
            }

        revenue_growth = round((info.get("revenueGrowth") or 0) * 100, 2)
        profit_margin = round((info.get("profitMargins") or 0) * 100, 2)
        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        high_52 = info.get("fiftyTwoWeekHigh")
        low_52 = info.get("fiftyTwoWeekLow")
        industry = info.get("industry")
        sector = info.get("sector")
        headquarters = ", ".join(
            p for p in [info.get("city"), info.get("state"), info.get("country")] if p
        ) or None
        ceo = info.get("_ceo")
        website = info.get("website")
        company_name = info.get("longName") or info.get("shortName") or company

        summary = (
            f"Symbol: {resolved_symbol} | Price: {current_price} | "
            f"Revenue Growth: {revenue_growth}% | Profit Margin: {profit_margin}% | "
            f"P/E: {pe_ratio} | Market Cap: {market_cap} | "
            f"52W Range: {low_52}-{high_52} | Sector: {sector} | Industry: {industry}"
            + (f" | CEO: {ceo}" if ceo else "")
        )

        return {
            "symbol": resolved_symbol,
            "publicly_traded": True,
            "summary": summary,
            "revenue_growth": revenue_growth,
            "profit_margin": profit_margin,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "current_price": current_price,
            "fifty_two_week_high": high_52,
            "fifty_two_week_low": low_52,
            "industry": industry,
            "sector": sector,
            "headquarters": headquarters,
            "ceo": ceo,
            "website": website,
            "company_name": company_name,
        }
    except Exception as e:
        logger.error(f"Financial agent error for {company} ({symbol}): {e}")
        return {
            "symbol": symbol, "publicly_traded": False,
            "summary": "Financial data unavailable due to an error fetching market data.",
            "industry": None, "sector": None, "headquarters": None,
        }


# ========================= NEWS ANALYSIS AGENT =========================
async def news_analysis_agent(company: str, news_chunks: List[str], has_real_news: bool = True) -> Dict:
    if not has_real_news:
        return {
            "sentiment": "Neutral",
            "score": 50,
            "positives": [],
            "negatives": [],
            "key_highlights": [
                "No live news data available — check SERPER_API_KEY / NEWS_API_KEY configuration. "
                "Score defaulted to neutral (50) rather than penalizing the company for missing coverage."
            ],
        }

    context = "\n\n".join(news_chunks[:5])
    prompt = f"""Analyze the following recent news about {company}:

{context}

Return ONLY valid JSON in exactly this shape, no extra text:
{{
  "sentiment": "Positive" | "Neutral" | "Negative",
  "score": <integer 0-100>,
  "positives": ["short point", "..."],
  "negatives": ["short point", "..."],
  "key_highlights": ["short point", "..."]
}}"""
    system = "You are an expert financial news analyst. Be precise, objective, and base your output only on the given text."
    result = await llm.generate(prompt, system)
    return result or {
        "sentiment": "Neutral",
        "score": 50,
        "positives": [],
        "negatives": [],
        "key_highlights": ["LLM analysis unavailable — check API keys."],
    }


# ========================= SWOT AGENT =========================
async def swot_agent(company: str, financial: Dict, news_chunks: List[str]) -> Dict:
    context = "\n\n".join(news_chunks[:5])
    prompt = f"""Based on this data about {company}, produce a SWOT analysis.

Financial data: {financial}
Recent news: {context}

Return ONLY valid JSON in exactly this shape:
{{
  "strengths": ["..."],
  "weaknesses": ["..."],
  "opportunities": ["..."],
  "threats": ["..."]
}}"""
    system = "You are a senior equity research analyst. Ground every point in the data given; do not invent facts."
    result = await llm.generate(prompt, system)
    return result or {
        "strengths": [], "weaknesses": [], "opportunities": [], "threats": [],
    }


# ========================= RISK AGENT =========================
async def risk_agent(company: str, financial: Dict, news_chunks: List[str]) -> Dict:
    context = "\n\n".join(news_chunks[:5])
    prompt = f"""Assess investment risk for {company} based on this data.

Financial data: {financial}
Recent news: {context}

Return ONLY valid JSON in exactly this shape:
{{
  "risk_score": <integer 0-100, where 100 = highest risk>,
  "risks": ["short risk factor 1", "short risk factor 2", "..."]
}}"""
    system = "You are a risk analyst. Identify concrete, specific risks (regulatory, competitive, valuation, execution, macro) grounded in the data given."
    result = await llm.generate(prompt, system)
    if not result:
        return {"risk_score": 50, "risks": ["Risk analysis unavailable — check API keys."]}
    return result


# ========================= DETERMINISTIC SCORING ENGINE =========================
def compute_rule_based_score(financial: Dict, news: Dict, risk: Dict) -> Dict:
    """
    Computes a deterministic baseline score from hard numbers so the LLM
    can't talk itself into INVEST despite bad fundamentals. The LLM is
    still used for narrative reasoning, but the score itself is math-driven.
    """
    if not financial.get("publicly_traded", True):
        return {"score": 0, "notes": ["Not publicly traded — scored as WATCH/PASS pending more info."]}

    score = 50
    notes = []

    revenue_growth = financial.get("revenue_growth")
    profit_margin = financial.get("profit_margin")
    pe_ratio = financial.get("pe_ratio")
    news_score = news.get("score", 50)
    risk_score = risk.get("risk_score", 50)

    if revenue_growth is not None:
        if revenue_growth > 30:
            score += 20; notes.append("Strong revenue growth >30% (+20)")
        elif revenue_growth > 15:
            score += 10; notes.append("Solid revenue growth >15% (+10)")
        elif revenue_growth > 5:
            score += 5; notes.append("Modest revenue growth >5% (+5)")
        elif revenue_growth < 0:
            score -= 15; notes.append("Declining revenue (-15)")

    if profit_margin is not None:
        if profit_margin > 25:
            score += 15; notes.append("Excellent profit margin >25% (+15)")
        elif profit_margin > 10:
            score += 10; notes.append("Healthy profit margin >10% (+10)")
        elif profit_margin > 0:
            score += 3; notes.append("Positive profit margin (+3)")
        elif profit_margin < 0:
            score -= 20; notes.append("Negative profit margin (-20)")

    if pe_ratio is not None and pe_ratio > 0 and pe_ratio > 60:
        score -= 5; notes.append("Very high valuation/P-E (-5)")

    news_adjustment = int(round((news_score - 50) * 0.30))
    score += news_adjustment
    notes.append(f"News sentiment adjustment ({news_score}/100) ({'+' if news_adjustment >= 0 else ''}{news_adjustment})")

    risk_adjustment = int(round(max(0, risk_score - 40) * 0.25))
    score -= risk_adjustment
    notes.append(f"Risk penalty ({risk_score}/100, baseline 40) (-{risk_adjustment})")

    score = max(0, min(100, score))
    return {"score": score, "notes": notes}


def _label_from_score(score: float, invest_at: float = 65, watch_at: float = 40) -> str:
    if score >= invest_at:
        return "INVEST"
    if score >= watch_at:
        return "WATCH"
    return "PASS"


def compute_committee_vote(financial: Dict, news: Dict, risk: Dict, swot: Dict, final_recommendation: str) -> Dict:
    """
    Each agent independently 'votes' based only on its own slice of data,
    so the final response shows real (if simple) multi-agent disagreement
    rather than one LLM call dressed up as four.
    """
    if not financial.get("publicly_traded", True):
        return {
            "financial_agent": "WATCH", "news_agent": "WATCH",
            "risk_agent": "WATCH", "swot_agent": "WATCH",
            "final_decision": "WATCH",
        }

    fin_score = 50
    rg, pm = financial.get("revenue_growth"), financial.get("profit_margin")
    if rg is not None:
        fin_score += 20 if rg > 30 else 10 if rg > 15 else 5 if rg > 5 else (-15 if rg < 0 else 0)
    if pm is not None:
        fin_score += 15 if pm > 25 else 10 if pm > 10 else 3 if pm > 0 else (-20 if pm < 0 else 0)
    financial_vote = _label_from_score(max(0, min(100, fin_score)))

    news_vote = _label_from_score(news.get("score", 50), invest_at=70, watch_at=45)

    risk_score = risk.get("risk_score", 50)
    risk_vote = "PASS" if risk_score > 70 else "WATCH" if risk_score > 40 else "INVEST"

    positives = len(swot.get("strengths", [])) + len(swot.get("opportunities", []))
    negatives = len(swot.get("weaknesses", [])) + len(swot.get("threats", []))
    if positives - negatives >= 2:
        swot_vote = "INVEST"
    elif negatives - positives >= 2:
        swot_vote = "PASS"
    else:
        swot_vote = "WATCH"

    return {
        "financial_agent": financial_vote,
        "news_agent": news_vote,
        "risk_agent": risk_vote,
        "swot_agent": swot_vote,
        "final_decision": final_recommendation,
    }


async def decision_agent(state: Dict) -> Dict:
    rule_result = state.get("rule_score", {"score": 50, "notes": []})
    rule_score = rule_result["score"]
    financial = state.get("financial", {})

    if not financial.get("publicly_traded", True):
        prompt = f"""{state['company']} is a privately held company (no public stock).
Financial data: {financial}
News analysis: {state.get('news')}

Return ONLY valid JSON:
{{
  "recommendation": "WATCH",
  "overall_score": 0,
  "confidence_score": <integer 0-100>,
  "executive_summary": "<note that it's private and cannot be invested in directly via public markets>",
  "bull_case": "<2-3 sentences on the private company's prospects>",
  "bear_case": "<2-3 sentences on risks>",
  "reasoning": "<note it is not publicly traded; recommend monitoring for IPO/funding news>",
  "confidence_reason": "<short>"
}}"""
        result = await llm.generate(prompt, "Return clean JSON only, no markdown.")
        result = result or {}
        result["overall_score"] = 0
        result.setdefault("recommendation", "WATCH")
        return result

    prompt = f"""You are a Senior Investment Analyst at an investment committee. A deterministic
scoring model has already computed an overall_score of {rule_score}/100 for {state['company']}
based on these factors: {rule_result['notes']}.

Use overall_score = {rule_score} EXACTLY as given — do not change it. Your job is to pick the
recommendation label consistent with that score and write the narrative reasoning.

Scoring guide: score >= 70 -> "INVEST", 40-69 -> "WATCH", < 40 -> "PASS".

Financial Data: {financial}
News Analysis: {state.get('news')}
SWOT: {state.get('swot')}
Risk Assessment: {state.get('risk')}

Return ONLY valid JSON in exactly this shape:
{{
  "recommendation": "INVEST" | "PASS" | "WATCH",
  "overall_score": {rule_score},
  "confidence_score": <integer 0-100>,
  "executive_summary": "<2-3 sentence summary>",
  "bull_case": "<2-3 sentences>",
  "bear_case": "<2-3 sentences>",
  "reasoning": "<3-5 sentences explaining the recommendation, referencing the specific factors that drove the score>",
  "confidence_reason": "<1-2 sentences on why this confidence level>"
}}"""
    result = await llm.generate(prompt, "You are an Investment Committee. Return clean JSON only, no markdown.")
    result = result or {}
    result["overall_score"] = rule_score
    if "recommendation" not in result or not result["recommendation"]:
        result["recommendation"] = "INVEST" if rule_score >= 70 else "WATCH" if rule_score >= 40 else "PASS"
    return result


# ========================= LANGGRAPH WORKFLOW =========================
class AgentState(TypedDict):
    company: str
    research: Dict
    financial: Dict
    news: Dict
    swot: Dict
    risk: Dict
    rule_score: Dict
    final_decision: Dict
    sources: List[str]


async def research_node(state: AgentState):
    research, financial = await asyncio.gather(
        research_agent(state["company"]),
        financial_agent(state["company"]),
    )
    return {**state, "research": research, "financial": financial, "sources": research.get("sources", [])}


async def analysis_node(state: AgentState):
    news, swot, risk = await asyncio.gather(
        news_analysis_agent(state["company"], state["research"].get("news_chunks", []),
                             state["research"].get("has_real_news", True)),
        swot_agent(state["company"], state["financial"], state["research"].get("news_chunks", [])),
        risk_agent(state["company"], state["financial"], state["research"].get("news_chunks", [])),
    )
    rule_score = compute_rule_based_score(state["financial"], news, risk)
    return {**state, "news": news, "swot": swot, "risk": risk, "rule_score": rule_score}


async def decision_node(state: AgentState):
    decision = await decision_agent(state)
    return {**state, "final_decision": decision}


def build_graph():
    wf = StateGraph(AgentState)
    wf.add_node("research", research_node)
    wf.add_node("analysis", analysis_node)
    wf.add_node("decision", decision_node)
    wf.set_entry_point("research")
    wf.add_edge("research", "analysis")
    wf.add_edge("analysis", "decision")
    wf.add_edge("decision", END)
    return wf.compile()


graph = build_graph()

# ========================= FASTAPI APP =========================
app = FastAPI(
    title="AI Investment Research Agent",
    version="3.1",
    description="Real-time news + financial analysis using LangGraph multi-agent workflow.",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "3.1",
        "providers": {
            "gemini": bool(GEMINI_API_KEY),
            "groq": bool(GROQ_API_KEY),
            "serper": bool(SERPER_API_KEY),
            "newsapi": bool(NEWS_API_KEY),
            "alpha_vantage": bool(ALPHA_VANTAGE_API_KEY),
            "finnhub": bool(FINNHUB_API_KEY),
            "fmp": bool(FMP_API_KEY),
        },
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalyzeRequest):
    company = request.company.strip()
    logger.info(f"Analyzing: {company}")

    try:
        result = await graph.ainvoke({
            "company": company,
            "research": {}, "financial": {}, "news": {}, "swot": {}, "risk": {}, "rule_score": {},
            "final_decision": {}, "sources": [],
        })

        dec = result.get("final_decision", {}) or {}
        financial = result.get("financial", {}) or {}
        news = result.get("news", {}) or {}
        swot = result.get("swot", {}) or {}
        risk = result.get("risk", {}) or {}
        committee_vote = compute_committee_vote(financial, news, risk, swot, dec.get("recommendation", "WATCH"))

        return AnalysisResponse(
            company=company,
            recommendation=dec.get("recommendation", "PASS"),
            overall_score=int(dec.get("overall_score", 50)),
            confidence_score=int(dec.get("confidence_score", 40)),
            confidence_reason=dec.get("confidence_reason", "Limited data — verify API keys are configured."),
            news_sentiment=news.get("sentiment", "Neutral"),
            executive_summary=dec.get("executive_summary", "Analysis completed with available data."),
            financial_analysis=financial.get("summary", ""),
            financial_data=FinancialSnapshot(**{k: v for k, v in financial.items() if k in FinancialSnapshot.model_fields}),
            risk_assessment=dec.get("reasoning", ""),
            swot=SWOT(**{k: v for k, v in swot.items() if k in SWOT.model_fields}),
            news_analysis=NewsAnalysis(**{k: v for k, v in news.items() if k in NewsAnalysis.model_fields}),
            risk=RiskAssessment(**{k: v for k, v in risk.items() if k in RiskAssessment.model_fields}),
            investment_committee=CommitteeVote(**committee_vote),
            bull_case=dec.get("bull_case", ""),
            bear_case=dec.get("bear_case", ""),
            reasoning=dec.get("reasoning", ""),
            sources=result.get("sources", []),
        )
    except Exception as e:
        logger.exception(f"Analysis failed for {company}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)