import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from google.genai.errors import APIError as GeminiAPIError
from openai import OpenAIError

load_dotenv(Path(__file__).with_name(".env"))

try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from agno.tools.duckduckgo import DuckDuckGoTools
except ImportError:  # pragma: no cover - graceful fallback
    Agent = None
    OpenAIChat = None
    DuckDuckGoTools = None

try:
    from agno.models.google import Gemini
except ImportError:  # pragma: no cover - optional provider dependency
    Gemini = None


st.set_page_config(page_title="ShopWise AI", page_icon="🛍️", layout="wide")

EXAMPLES = [
    "I need a smartphone under ₹20,000 for my mother. Good battery, simple UI and reliable performance are important.",
    "Wireless headphones under ₹5,000 for office meetings. Good microphone and comfort.",
    "Laptop under ₹60,000 for an MBA student. Good battery, portability and productivity performance.",
]

CATEGORY_KEYWORDS = {
    "Smartphone": ["smartphone", "phone", "mobile", "android", "iphone"],
    "Laptop": ["laptop", "notebook", "ultrabook"],
    "Headphones": ["headphone", "earbuds", "earphones", "headset"],
    "Smartwatch": ["smartwatch", "watch", "fitness watch"],
    "Monitor": ["monitor", "display", "screen"],
    "Keyboard": ["keyboard", "mechanical keyboard"],
}

BUDGET_OPTIONS = ["Any", "Under ₹5,000", "Under ₹10,000", "Under ₹25,000", "Custom"]


class ShoppingBrief(BaseModel):
    category: str = "Not specified"
    budget: str = "Not specified"
    primary_use: str = "Not specified"
    priority: str = "Not specified"
    brand_preferences: str = "Not specified"
    constraints: str = "Not specified"


class ProductRecommendation(BaseModel):
    name: str
    price: str = "Not verified"
    why: str = "Information unavailable."
    strengths: List[str] = Field(default_factory=list)
    tradeoff: str = "Information unavailable."
    best_for: str = "Not specified"


class ResearchSource(BaseModel):
    title: str
    url: str


class ShoppingResearch(BaseModel):
    brief: ShoppingBrief
    products: List[ProductRecommendation]
    decision_matrix: Dict[str, List[str]] = Field(default_factory=dict)
    trade_off: str = "Information unavailable."
    sources: List[ResearchSource] = Field(default_factory=list)


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def detect_category(query: str) -> str:
    lower_query = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lower_query for keyword in keywords):
            return category
    return "Not specified"


def normalize_budget(value: str, multiplier: int = 1) -> str:
    cleaned = value.replace(",", "")
    if not cleaned:
        return "Not specified"
    num = float(cleaned) * multiplier
    return f"Under ₹{num:,.0f}"


def extract_budget(query: str, chosen_budget: str) -> str:
    amount_pattern = r"(?:under|below|within|budget(?:\s+of)?|up to|upto)\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|lakhs?|l)?\b"
    match = re.search(amount_pattern, query, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"₹\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|lakhs?|l)?\b", query, flags=re.IGNORECASE)
    if match:
        suffix = (match.group(2) or "").lower()
        multiplier = 100_000 if suffix in ("l", "lakh", "lakhs") else 1_000 if suffix in ("k", "thousand") else 1
        return normalize_budget(match.group(1), multiplier)
    if chosen_budget in ["Under ₹5,000", "Under ₹10,000", "Under ₹25,000"]:
        return chosen_budget
    return "Not specified"


def extract_target_user(query: str) -> str:
    lower_query = query.lower()
    if any(term in lower_query for term in ["mother", "mom", "parents", "parent"]):
        return "Parent / everyday use"
    if any(term in lower_query for term in ["student", "mba", "college", "school"]):
        return "Student / productivity"
    if any(term in lower_query for term in ["office", "work", "professional", "meetings"]):
        return "Office / work use"
    if any(term in lower_query for term in ["gift", "family", "household"]):
        return "Gift / family use"
    return "Not specified"


def extract_priority(query: str) -> str:
    priorities = []
    for item in ["battery", "camera", "performance", "comfort", "microphone", "portability", "reliability", "simple ui", "value"]:
        if item in query.lower():
            priorities.append(item.capitalize())
    if not priorities:
        return "Not specified"
    return ", ".join(priorities[:4])


def parse_agent_response(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("No response content received.")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        first_bracket = text.find("{")
        last_bracket = text.rfind("}")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            candidate = text[first_bracket : last_bracket + 1]
            parsed = json.loads(candidate)
        else:
            raise ValueError("Response could not be parsed as JSON.")
    if not isinstance(parsed, dict):
        raise ValueError("The research response must be a JSON object.")
    return parsed


def provider_error(content: Any, status: Any) -> tuple[str, int | None] | None:
    status_value = str(getattr(status, "value", status) or "").lower()
    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            parsed_content = None
        if isinstance(parsed_content, dict):
            nested_error = parsed_content.get("error")
            if nested_error:
                return provider_error({"error": nested_error}, status)
    if isinstance(content, dict) and content.get("error"):
        error = content["error"]
        if isinstance(error, dict):
            message = str(error.get("message") or error)
            code = error.get("code")
            try:
                error_code = int(code) if code is not None else None
            except (TypeError, ValueError):
                error_code = None
            return message, error_code
        return str(error), None
    if status_value == "error":
        return str(content or "The model request failed without an error message."), None
    return None


def add_grounding_sources(result: Dict[str, Any], citations: Any) -> None:
    if citations is None:
        return
    if isinstance(citations, dict):
        citation_urls = citations.get("urls", [])
    else:
        citation_urls = getattr(citations, "urls", [])

    if not isinstance(citation_urls, list):
        return

    sources = result.setdefault("sources", [])
    if not isinstance(sources, list):
        sources = []
        result["sources"] = sources
    known_urls = {
        source.get("url")
        for source in sources
        if isinstance(source, dict) and source.get("url")
    }
    for citation in citation_urls or []:
        if isinstance(citation, dict):
            url = citation.get("url")
            title = citation.get("title")
        else:
            url = getattr(citation, "url", None)
            title = getattr(citation, "title", None)
        if url and url not in known_urls:
            sources.append({"url": url, "title": title or url})
            known_urls.add(url)


def do_live_research(query: str, brief: Dict[str, str]) -> Dict[str, Any]:
    if not Agent:
        raise RuntimeError("Agno is unavailable. Install the project dependencies and try again.")

    provider = os.getenv("AI_PROVIDER", "gemini").strip().lower()
    gemini_models: List[str] = []
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add your key to the project .env file and restart the app.")
        if Gemini is None:
            raise RuntimeError("Gemini support is unavailable. Install the project dependencies and try again.")
        primary_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
        fallback_models = os.getenv("GEMINI_FALLBACK_MODEL", "")
        gemini_models = [primary_model]
        gemini_models.extend(
            fallback_model.strip()
            for fallback_model in fallback_models.split(",")
            if fallback_model.strip() and fallback_model.strip() not in gemini_models
        )
        model = None
    elif provider == "openai":
        if OpenAIChat is None:
            raise RuntimeError("OpenAI support is unavailable. Install the project dependencies and try again.")
        if DuckDuckGoTools is None:
            raise RuntimeError("Agno web search is unavailable. Install the project dependencies and try again.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add your key to the project .env file and restart the app.")
        model = OpenAIChat(
            id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.2,
        )
    else:
        raise ValueError("AI_PROVIDER must be set to 'gemini' or 'openai'.")

    prompt = (
        "Research a product shortlist for this shopper request. "
        "Use the provided shopping brief and query to research current product options. "
        "Provide up to 3 options, with verified Indian market prices only when search results support them. "
        "Include the keys and fields required by the response schema. Never use a different top-level key instead of 'products'.\n\n"
        f"User request: {query}\n"
        f"Shopping brief: {json.dumps(brief, ensure_ascii=False)}\n"
    )
    model_attempts = (
        [Gemini(id=model_id, api_key=api_key, search=True) for model_id in gemini_models]
        if provider == "gemini"
        else [model]
    )
    last_error: RuntimeError | None = None

    for index, attempt_model in enumerate(model_attempts):
        agent = Agent(
            model=attempt_model,
            tools=(
                []
                if provider == "gemini"
                else [DuckDuckGoTools(enable_news=False, fixed_max_results=5)]
            ),
            output_schema=ShoppingResearch,
            structured_outputs=True,
            instructions=[
                "You are a careful product research assistant for Indian shoppers.",
                "Use web search for current product facts. Never guess or fabricate prices, ratings, specifications, availability, citations, or URLs.",
                "If a fact is uncertain, use 'Not verified' or 'Information unavailable'.",
                "Fill the required structured response. Return at least two distinct products when the available research supports it.",
                "Use 'Not specified' in brief fields when the shopper did not provide that information.",
                "For decision_matrix, use 'Criteria' for the relevant criterion labels and one qualitative rating array per product, in matching order. Use Strong, Good, Average, or Not verified.",
                "Only include source URLs actually returned by web search.",
                "Keep the output compact and return up to three distinct products.",
            ],
            markdown=False,
        )
        try:
            run_response = agent.run(prompt)
        except GeminiAPIError as exc:
            code = getattr(exc, "code", None)
            if provider == "gemini" and code in (429, 503) and index + 1 < len(model_attempts):
                last_error = RuntimeError(f"Gemini model {gemini_models[index]} is temporarily unavailable: {exc}")
                continue
            if provider == "gemini" and code == 429:
                raise RuntimeError(
                    "Gemini API quota/rate limit was reached for the configured key. "
                    "Check Google AI Studio's usage and billing limits, wait for quota to reset, "
                    f"or use a key with available quota. Details: {exc}"
                ) from exc
            if provider == "gemini" and code == 503:
                raise RuntimeError(
                    f"Gemini is temporarily unavailable or overloaded: {exc}"
                ) from exc
            raise

        content = getattr(run_response, "content", None)
        error = provider_error(content, getattr(run_response, "status", None))
        if error:
            message, code = error
            if provider == "gemini" and code in (429, 503) and index + 1 < len(model_attempts):
                last_error = RuntimeError(f"Gemini model {gemini_models[index]} is temporarily unavailable: {message}")
                continue
            provider_name = "Gemini" if provider == "gemini" else "OpenAI"
            if provider == "gemini" and code == 429:
                raise RuntimeError(
                    "Gemini API quota/rate limit was reached for the configured key. "
                    "Check Google AI Studio's usage and billing limits, wait for quota to reset, "
                    f"or use a key with available quota. Details: {message}"
                )
            if provider == "gemini" and code == 503:
                raise RuntimeError(
                    "Gemini is temporarily unavailable or overloaded. Please retry later. "
                    f"Details: {message}"
                )
            raise RuntimeError(f"{provider_name} request failed: {message}")
        if content is None:
            raise RuntimeError("The AI provider returned no response content.")
        if isinstance(content, ShoppingResearch):
            result = content.model_dump()
        elif isinstance(content, BaseModel):
            result = content.model_dump()
        elif isinstance(content, dict):
            result = content
        else:
            result = parse_agent_response(str(content))
        result = normalize_research_result(result)
        if provider == "gemini":
            add_grounding_sources(result, getattr(run_response, "citations", None))
        if not isinstance(result.get("products"), list):
            returned_keys = ", ".join(sorted(str(key) for key in result))
            raise ValueError(
                "Gemini returned an unexpected response structure; the required 'products' list is missing. "
                f"Returned fields: {returned_keys or 'none'}. Please retry."
            )
        result["products"] = [
            product for product in result["products"]
            if isinstance(product, dict) and product.get("name")
        ]
        if not result["products"]:
            raise ValueError("No product recommendations were returned. Try adding more detail to your request.")
        if provider == "gemini" and index > 0:
            result["research_model"] = gemini_models[index]
        return result

    if last_error is not None:
        raise last_error
    raise RuntimeError("The AI provider did not return a research result.")


def normalize_research_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common model response aliases to the dashboard's expected schema."""
    normalized = dict(result)
    if "products" not in normalized:
        for alias in ("recommendations", "product_recommendations", "shortlist", "items"):
            if isinstance(normalized.get(alias), list):
                normalized["products"] = normalized[alias]
                break

    brief = normalized.get("brief")
    if isinstance(brief, dict):
        aliases = {
            "category": "Category",
            "budget": "Budget",
            "primary_use": "Primary use",
            "primary use": "Primary use",
            "target_user": "Primary use",
            "priority": "Priority",
            "priorities": "Priority",
            "brand_preferences": "Brand preferences",
            "brand preferences": "Brand preferences",
            "constraints": "Constraints",
        }
        normalized["brief"] = {
            aliases.get(str(key).lower(), key): value
            for key, value in brief.items()
        }

    for product in normalized.get("products", []):
        if isinstance(product, dict):
            if "tradeoff" not in product and "trade_off" in product:
                product["tradeoff"] = product["trade_off"]
            if "strengths" not in product and "key_strengths" in product:
                product["strengths"] = product["key_strengths"]
    return normalized


def render_header() -> None:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">🛍️</div>
            <div>
                <div class="brand-title">ShopWise AI</div>
                <div class="brand-subtitle">Research smarter. Compare better. Buy with confidence.</div>
            </div>
        </div>
        <div class="badge-row">
            <span class="pill">AI Shopping Research Assistant</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-box">
            <h1>What are you looking to buy?</h1>
            <p>Tell us your budget, use case and preferences. ShopWise AI will research the market and build a personalized shortlist.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_buttons() -> None:
    labels = ["Smartphone for parents", "Work headphones", "Student laptop"]
    cols = st.columns(3)
    for idx, example in enumerate(EXAMPLES):
        with cols[idx]:
            st.caption(example)
            if st.button(labels[idx], key=f"example-{idx}"):
                st.session_state["_pending_query"] = example
                st.session_state.pop("research_result", None)
                st.rerun()


def render_query_form() -> bool:
    if "budget_choice" not in st.session_state:
        st.session_state["budget_choice"] = "Any"

    with st.form("shopping_form"):
        st.text_area(
            "Shopping request",
            placeholder="I need wireless headphones under ₹5,000 for office work. Good microphone, comfort and battery life are important.",
            key="query_input",
            height=160,
        )

        st.caption("Budget")
        st.selectbox(
            "Budget",
            BUDGET_OPTIONS,
            index=BUDGET_OPTIONS.index(st.session_state.get("budget_choice", "Any")),
            key="budget_choice",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("🔎 Research Products", use_container_width=True)

    st.caption("Category examples")
    quick_categories = [
        "📱 Smartphones",
        "💻 Laptops",
        "🎧 Headphones",
        "⌨️ Keyboards",
        "🖥️ Monitors",
        "⌚ Smartwatches",
    ]
    category_cols = st.columns(6)
    for idx, category in enumerate(quick_categories):
        with category_cols[idx]:
            if st.button(category, key=f"cat-{idx}"):
                chosen_budget = st.session_state.get("budget_choice", "Any")
                st.session_state["_pending_query"] = (
                    f"{category.replace('📱 ', '').replace('💻 ', '').replace('🎧 ', '').replace('⌨️ ', '').replace('🖥️ ', '').replace('⌚ ', '')} "
                    f"under {chosen_budget if chosen_budget != 'Any' else '₹25,000'} for everyday use."
                )
                st.session_state.pop("research_result", None)
                st.rerun()

    return submitted


def render_brief(brief: Dict[str, Any]) -> None:
    st.subheader("Your Shopping Brief")
    info = [
        ("Category", brief.get("Category", "Not specified")),
        ("Budget", brief.get("Budget", "Not specified")),
        ("Primary use", brief.get("Primary use", "Not specified")),
        ("Priority", brief.get("Priority", "Not specified")),
        ("Brand preferences", brief.get("Brand preferences", "Not specified")),
        ("Constraints", brief.get("Constraints", "Not specified")),
    ]

    cols = st.columns(2)
    for idx, (label, value) in enumerate(info):
        with cols[idx % 2]:
            with st.container(border=True):
                st.caption(label)
                st.write(value or "Not specified")


def render_product_cards(products: List[Dict[str, Any]]) -> None:
    st.subheader("Top product matches")
    for product in products:
        with st.container(border=True):
            name_col, price_col = st.columns([3, 1])
            with name_col:
                st.markdown(f"### {product.get('name', 'Product')}")
            with price_col:
                st.metric("Approx. price", product.get("price", "Not verified"))
            st.markdown("**Why it matches**")
            st.write(product.get("why", "Information unavailable."))
            strengths = ensure_list(product.get("strengths"))
            if strengths:
                st.markdown("**Key strengths**")
                for strength in strengths:
                    st.markdown(f"- {strength}")
            st.markdown("**Trade-off**")
            st.write(product.get("tradeoff", "Information unavailable."))
            st.markdown("**Best for**")
            st.write(product.get("best_for", "Not specified"))


def render_comparison_table(products: List[Dict[str, Any]]) -> None:
    st.subheader("Side-by-Side Comparison")
    comparison = [
        {
            "Product": product.get("name", "Not specified"),
            "Price": product.get("price", "Not verified"),
            "Key Strength": (ensure_list(product.get("strengths"))[0] if ensure_list(product.get("strengths")) else "Not specified"),
            "Best For": product.get("best_for", "Not specified"),
            "Main Trade-off": product.get("tradeoff", "Information unavailable."),
        }
        for product in products
    ]
    st.table(pd.DataFrame(comparison))


def render_decision_matrix(matrix: Dict[str, Any], products: List[Dict[str, Any]]) -> None:
    st.subheader("What matters most?")
    st.caption("AI-assisted qualitative comparisons based on researched product characteristics.")
    if not isinstance(matrix, dict):
        st.info("No qualitative decision matrix was returned for this research.")
        return
    criteria = ensure_list(matrix.get("Criteria"))
    if not criteria:
        st.info("No qualitative decision matrix was returned for this research.")
        return

    table: Dict[str, List[str]] = {"Criteria": criteria}
    for product in products:
        name = str(product.get("name", "Product"))
        ratings = ensure_list(matrix.get(name))
        if len(ratings) != len(criteria):
            ratings = ["Not verified"] * len(criteria)
        table[name] = ratings
    st.table(pd.DataFrame(table))


def render_tradeoff(trade_off_text: str) -> None:
    st.subheader("The Trade-Off")
    st.info(trade_off_text or "Information unavailable.")


def render_final_shortlist(products: List[Dict[str, Any]]) -> None:
    if not products:
        return
    best = products[0]
    alternative = products[1] if len(products) > 1 else None

    st.subheader("🎯 Your Shortlist")
    with st.container(border=True):
        st.markdown("**Best Match**")
        st.markdown(f"### {best.get('name', 'Product')}")
        st.markdown(
            f"Based on the requirements you provided, **{best.get('name', 'Product')}** appears to be the closest fit because "
            f"{best.get('why', 'its researched characteristics align with the stated priorities.')}"
        )
    if alternative:
        with st.container(border=True):
            st.markdown("**Alternative**")
            st.markdown(f"### {alternative.get('name', 'Alternative product')}")
            st.write(
                f"This may suit someone who prioritises {alternative.get('best_for', 'a different balance of trade-offs')}."
            )


def render_sources(sources: List[Any]) -> None:
    st.subheader("🔗 Research Sources")
    rendered = False
    if isinstance(sources, str):
        sources = [sources]
    for source in sources:
        if isinstance(source, dict):
            url = str(source.get("url", "")).strip()
            title = str(source.get("title") or url).strip()
        else:
            url = str(source).strip()
            title = url
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            st.link_button(title, url)
            rendered = True
    if not rendered:
        st.caption("No verifiable source URLs were returned.")


def render_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .brand-row {
                display: flex; align-items: center; gap: 16px; margin-bottom: 12px;
            }
            .brand-mark {
                width: 52px; height: 52px; border-radius: 16px; background: #eff6ff; display: flex; align-items: center; justify-content: center; font-size: 28px;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
            }
            .brand-title {
                font-size: 2rem; font-weight: 700; letter-spacing: -0.04em; color: #0f172a;
            }
            .brand-subtitle {
                color: #475569; font-size: 0.92rem;
            }
            .badge-row {
                margin-bottom: 1.5rem;
            }
            .pill {
                display: inline-block; padding: 7px 12px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 0.8rem; font-weight: 600;
            }
            .hero-box {
                margin: 0 0 1.5rem 0; padding: 1.5rem 1.5rem 0.75rem 1.5rem; border-radius: 22px; background: linear-gradient(135deg, #f8fafc, #eef2ff);
                border: 1px solid rgba(148, 163, 184, 0.2);
            }
            .hero-box h1 {
                margin: 0 0 0.5rem 0; font-size: 2.5rem; line-height: 1.1; letter-spacing: -0.05em; color: #0f172a;
            }
            .hero-box p {
                margin: 0; color: #475569; font-size: 1.05rem;
            }
            .stButton > button {
                border-radius: 12px; font-weight: 600; width: 100%;
            }
            .stTextInput input, .stTextArea textarea, .stSelectbox select {
                border-radius: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    pending_query = st.session_state.pop("_pending_query", None)
    if pending_query is not None:
        st.session_state["query_input"] = pending_query

    render_styles()
    render_header()
    render_hero()
    st.markdown("### Try an example")
    render_example_buttons()

    submitted = render_query_form()
    query = st.session_state.get("query_input", "")

    if submitted:
        if not query.strip():
            st.warning("Please describe what you want to buy first.")
            st.session_state.pop("research_result", None)
        else:
            brief = {
                "Category": detect_category(query),
                "Budget": extract_budget(query, st.session_state.get("budget_choice", "Any")),
                "Primary use": extract_target_user(query),
                "Priority": extract_priority(query),
                "Brand preferences": "Not specified",
                "Constraints": "Not specified",
            }
            try:
                with st.spinner("Researching products using web search and your selected AI provider..."):
                    result = do_live_research(query, brief)
                returned_brief = result.get("brief")
                if isinstance(returned_brief, dict):
                    brief.update(
                        {
                            key: returned_brief.get(key) or value
                            for key, value in brief.items()
                        }
                    )
                st.session_state["research_result"] = {
                    "brief": brief,
                    "result": result,
                    "query": query,
                }
            except (RuntimeError, ValueError, GeminiAPIError, OpenAIError, DDGSException, RatelimitException, TimeoutException) as exc:
                st.session_state.pop("research_result", None)
                st.error(f"Product research failed: {exc}")
                if isinstance(exc, RuntimeError) and "API_KEY is missing" in str(exc):
                    provider = os.getenv("AI_PROVIDER", "gemini").strip().lower()
                    key_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
                    st.info(f"Copy `.env.example` to `.env`, add your key to `{key_name}`, save, and restart Streamlit.")

    research_result = st.session_state.get("research_result")
    if research_result:
        result = research_result["result"]
        products = result["products"]
        render_brief(research_result["brief"])
        render_product_cards(products)
        render_comparison_table(products)
        render_decision_matrix(result.get("decision_matrix", {}), products)
        render_tradeoff(result.get("trade_off", "Information unavailable."))
        render_final_shortlist(products)
        render_sources(result.get("sources", []))
    elif not submitted:
        st.info("Describe what you want to buy, then select Research Products to start live research.")


if __name__ == "__main__":
    main()
