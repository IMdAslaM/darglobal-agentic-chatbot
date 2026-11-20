"""
rag.py - Retrieval-Augmented Generation (RAG) & Agentic Chat Engine.

Connects to OpenRouter (https://openrouter.ai) using the OpenAI Python SDK.
Retrieves and ranks properties from DarGlobal and Wasalt using a hybrid
keyword + boost scoring engine, then generates conversational AI responses.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("property_rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "properties.json")

# ---------------------------------------------------------------------------
# Free models on OpenRouter (updated Nov 2026 – live-verified via /api/v1/models)
# openrouter/free = auto-routes to any currently available free model (most reliable)
# ---------------------------------------------------------------------------
DEFAULT_MODELS: List[str] = [
    "openrouter/free",                          # Auto-router — always picks an available free model
    "google/gemma-4-31b-it:free",               # Google Gemma 4 31B
    "nvidia/nemotron-3-ultra-550b-a55b:free",   # Nvidia Nemotron 550B
    "nvidia/nemotron-3.5-lightning:free",        # Nvidia Nemotron Lightning
    "google/gemma-4-26b-a4b-it:free",           # Google Gemma 4 26B
    "particle-ai/particle-1:free",              # Particle AI
    "liquid/lfm-3-1.7b:free",                   # Liquid Foundation Model
]

# ---------------------------------------------------------------------------
# Location aliases – map common spellings / abbreviations to canonical forms
# that appear in our dataset fields
# ---------------------------------------------------------------------------
LOCATION_ALIASES: Dict[str, List[str]] = {
    "palm jumeirah": ["palm", "jumeirah", "palm jumeirah", "dubai palm"],
    "dubai marina": ["marina", "dubai marina", "jbr"],
    "business bay": ["business bay", "canal"],
    "downtown dubai": ["downtown", "burj khalifa", "dubai mall"],
    "jvc": ["jumeirah village", "jumeirah village circle", "jvc"],
    "rak": ["ras al khaimah", "al marjan", "marjan island"],
    "riyadh": ["riyadh", "al malqa", "al narjis", "hittin", "al yasmin", "al janadriyah"],
    "jeddah": ["jeddah", "al shati", "obhur", "al mohammedia", "red sea", "corniche"],
    "muscat": ["muscat", "oman", "yiti", "aida"],
    "london": ["london", "mayfair", "old park lane"],
    "marbella": ["marbella", "spain", "costa del sol", "benahav"],
    "qatar": ["qatar", "doha", "lusail", "qetaifan"],
}

# Common typo corrections
TYPO_MAP: Dict[str, str] = {
    "pal jumeirah": "palm jumeirah",
    "palm jumirah": "palm jumeirah",
    "palm jumeriah": "palm jumeirah",
    "dubaii": "dubai",
    "ryiadh": "riyadh",
    "riyad": "riyadh",
    "jeddah": "jeddah",
    "jedah": "jeddah",
    "muscat": "muscat",
    "abu dabhi": "abu dhabi",
    "abu dhabi": "abu dhabi",
}

_BRAND_TERMS = {
    "pagani", "lamborghini", "missoni", "aston martin", "fendi",
    "trump", "elie saab", "marriott", "bentley", "armani", "w hotel",
}
_TYPE_TERMS = {
    "villa", "apartment", "penthouse", "mansion", "townhouse",
    "suite", "studio", "residence", "flat",
}
_STOP_WORDS = {
    "the", "a", "an", "in", "of", "for", "and", "or", "is", "are",
    "with", "to", "at", "on", "by", "me", "my", "i", "what", "how",
    "show", "find", "tell", "about", "available", "properties", "list",
    "looking", "want", "need", "please", "can", "you", "give", "get",
    "some", "any", "which", "where", "that", "this", "it", "do",
}


def _correct_query(query: str) -> str:
    """Apply typo corrections to the query."""
    q = query.lower().strip()
    for wrong, right in TYPO_MAP.items():
        q = q.replace(wrong, right)
    return q


class PropertyRAG:
    """
    Retrieval-Augmented Generation engine for real estate Q&A.

    Search scoring:
      title match              +6.0 / term
      district / city match    +5.0 / term
      location alias match     +8.0 flat (when user clearly names a location)
      property type match      +3.5 / term
      key features match       +2.5 / term
      description match        +1.5 / term
      portal explicit mention  +8.0 flat
      brand match              +10.0 flat
    """

    def __init__(self, data_path: str = DATA_PATH) -> None:
        self.data_path = data_path
        self.properties: List[Dict[str, Any]] = self._load_data()
        self.openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.primary_model: str = os.getenv("OPENROUTER_MODEL", DEFAULT_MODELS[0])

        self.client: Optional[OpenAI] = None
        if self.openrouter_api_key and not self.openrouter_api_key.startswith("sk-or-v1-your"):
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
                default_headers={
                    "HTTP-Referer": "http://localhost:7860",
                    "X-Title": "DarGlobal & Wasalt Real Estate AI",
                },
            )
            logger.info(f"OpenRouter client initialised. Primary model: {self.primary_model}")
        else:
            logger.warning("OPENROUTER_API_KEY not configured — offline fallback mode active.")

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def _load_data(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    logger.info(f"RAG Engine loaded {len(data)} properties.")
                    return data if isinstance(data, list) else []
            except Exception as exc:
                logger.error(f"Failed to load property dataset: {exc}")
        else:
            logger.warning(f"Dataset not found at {self.data_path}. Run scraper.py first.")
        return []

    def reload_data(self) -> int:
        self.properties = self._load_data()
        return len(self.properties)

    # ------------------------------------------------------------------
    # Search / retrieval
    # ------------------------------------------------------------------

    def _detect_location_intent(self, query: str) -> Optional[str]:
        """
        Detect if the user is asking about a specific location.
        Returns the canonical location key if detected.
        """
        q = query.lower()
        for canonical, aliases in LOCATION_ALIASES.items():
            for alias in aliases:
                if alias in q:
                    return canonical
        return None

    def search_properties(
        self,
        query: str,
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: keyword scoring + location alias boost + brand boost.
        Returns top_k best matches.
        """
        if not self.properties:
            return []

        # Apply typo correction
        corrected = _correct_query(query)
        query_lower = query.lower()

        terms = [
            t for t in re.findall(r"\w+", corrected)
            if t not in _STOP_WORDS and len(t) > 2
        ]

        # Detect explicit portal mention
        has_darglobal = any(x in query_lower for x in ("darglobal", "dar global", "dar-global"))
        has_wasalt = any(x in query_lower for x in ("wasalt", "وصلت"))

        # Detect location intent
        location_intent = self._detect_location_intent(corrected)

        scored: List[Tuple[float, Dict[str, Any]]] = []

        for prop in self.properties:
            # --- Hard filters ---
            if filters:
                portal_f = (filters.get("portal") or "").lower()
                city_f = (filters.get("city") or "").lower()
                type_f = (filters.get("property_type") or "").lower()
                if portal_f and portal_f != prop.get("developer_or_portal", "").lower():
                    continue
                if city_f and city_f not in prop.get("city", "").lower():
                    continue
                if type_f and type_f not in prop.get("property_type", "").lower():
                    continue

            score = 0.0
            portal = prop.get("developer_or_portal", "").lower()
            city = prop.get("city", "").lower()
            district = prop.get("district", "").lower()
            p_type = prop.get("property_type", "").lower()
            title = prop.get("title", "").lower()
            desc = prop.get("description", "").lower()
            features = " ".join(prop.get("key_features", [])).lower()
            combined = f"{title} {portal} {city} {district} {p_type} {desc} {features}"

            # --- Location alias boost (most important for location queries) ---
            if location_intent:
                aliases = LOCATION_ALIASES.get(location_intent, [])
                for alias in aliases:
                    if alias in district or alias in city or alias in title or alias in desc:
                        score += 8.0
                        break

            # --- Portal boost ---
            if has_darglobal and "darglobal" in portal:
                score += 8.0
            if has_wasalt and "wasalt" in portal:
                score += 8.0

            # --- Term scoring ---
            for term in terms:
                if term in title:
                    score += 6.0
                elif term in district or term in city:
                    score += 5.0
                elif term in p_type:
                    score += 3.5
                elif term in features:
                    score += 2.5
                elif term in desc:
                    score += 1.5

            # --- Brand boost ---
            for brand in _BRAND_TERMS:
                if brand in corrected and brand in combined:
                    score += 10.0
                    break

            # --- Property type boost ---
            for ptype in _TYPE_TERMS:
                if ptype in corrected and ptype in p_type:
                    score += 4.0
                    break

            if score > 0:
                scored.append((score, prop))

        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            results = [item[1] for item in scored[:top_k]]
            # If there's a strong location intent, filter out results that don't
            # match the location at all (score from alias boost only, not generic)
            if location_intent:
                aliases = LOCATION_ALIASES.get(location_intent, [])
                location_matched = [
                    item for item in scored
                    if any(
                        alias in f"{item[1].get('district','')} {item[1].get('city','')} {item[1].get('title','')}".lower()
                        for alias in aliases
                    )
                ]
                if location_matched:
                    return [item[1] for item in location_matched[:top_k]]
                # No exact location match — return all scored but warn via empty location match
            return results

        # Generic query fallback — return first top_k
        return self.properties[:top_k]

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context_prompt(self, properties: List[Dict[str, Any]]) -> str:
        if not properties:
            return "No matching properties found in the database for this query."

        blocks: List[str] = []
        for i, p in enumerate(properties, 1):
            features = ", ".join(p.get("key_features", []))
            usd = p.get("price_usd", 0)
            usd_str = f"~${usd:,}" if usd else "Contact for pricing"
            block = (
                f"Property #{i}:\n"
                f"  Title: {p.get('title')}\n"
                f"  Portal / Developer: {p.get('developer_or_portal')}\n"
                f"  Location: {p.get('district')}, {p.get('city')}, {p.get('country')}\n"
                f"  Property Type: {p.get('property_type')}\n"
                f"  Price: {p.get('price')} (USD {usd_str})\n"
                f"  Bedrooms: {p.get('bedrooms')} | Bathrooms: {p.get('bathrooms')}\n"
                f"  Area: {p.get('area_sqft')}\n"
                f"  Status: {p.get('status')}\n"
                f"  Description: {p.get('description')}\n"
                f"  Key Features: {features}\n"
                f"  Direct Link: {p.get('source_url')}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Conversational offline fallback (no API key / all models failed)
    # ------------------------------------------------------------------

    def _generate_conversational_fallback(
        self,
        query: str,
        matched: List[Dict[str, Any]],
        corrected_query: str,
    ) -> str:
        """
        Generate a natural, conversational response — not a robotic template.
        Mimics the style of ChatGPT/Claude responses.
        """
        location_intent = self._detect_location_intent(corrected_query)

        # Check if the matched properties actually match the user's location intent
        relevant = []
        if location_intent:
            aliases = LOCATION_ALIASES.get(location_intent, [])
            for p in matched:
                combined = f"{p.get('district','')} {p.get('city','')} {p.get('title','')} {p.get('description','')}".lower()
                if any(alias in combined for alias in aliases):
                    relevant.append(p)
        else:
            relevant = matched

        # --- No relevant results for the specific location ---
        if location_intent and not relevant:
            location_display = location_intent.title()
            # Suggest nearby alternatives
            alternatives = [p for p in matched if p.get("city", "").lower() == "dubai"] or matched[:2]
            response = (
                f"I appreciate your interest in **{location_display}** — it's one of Dubai's most iconic addresses! "
                f"However, our current curated portfolio doesn't include active listings specifically on {location_display} at this moment.\n\n"
                f"That said, I can offer you some excellent alternatives in Dubai that share similar prestige and lifestyle appeal:\n\n"
            )
            for p in alternatives[:2]:
                usd = p.get("price_usd", 0)
                usd_str = f" (~${usd:,})" if usd else ""
                response += (
                    f"### {p.get('title')}\n"
                    f"📍 **{p.get('district')}, {p.get('city')}** | {p.get('property_type')}\n"
                    f"💰 **{p.get('price')}**{usd_str}\n"
                    f"🛏 {p.get('bedrooms')} · 🚿 {p.get('bathrooms')} baths · 📐 {p.get('area_sqft')}\n\n"
                    f"{p.get('description', '')[:200]}...\n\n"
                    f"✨ **Highlights:** {', '.join((p.get('key_features') or [])[:3])}\n"
                    f"🔗 [View Full Details]({p.get('source_url')})\n\n---\n\n"
                )
            response += (
                "Would you like me to search for a broader range of Dubai luxury properties, "
                "or would you like details on any of the above options? I'm also happy to explore "
                "off-plan opportunities near Palm Jumeirah."
            )
            return response

        if not relevant:
            return (
                "I'd love to help with your property search! Currently, our database covers:\n\n"
                "🏰 **DarGlobal** — Luxury branded residences in Dubai, London, Muscat, Marbella, and Qatar\n"
                "🏢 **Wasalt** — Verified properties across Riyadh, Jeddah, and Saudi Arabia\n\n"
                "Could you tell me more about what you're looking for? For example:\n"
                "- A specific city or neighbourhood?\n"
                "- Your budget range?\n"
                "- Property type (villa, apartment, penthouse)?\n\n"
                "I'll find the best matches for you right away."
            )

        # --- Good relevant matches found ---
        props_to_show = relevant[:3]
        location_phrase = f" in **{location_intent.title()}**" if location_intent else ""

        intro_options = [
            f"Great news! I found **{len(props_to_show)} premium propert{'y' if len(props_to_show)==1 else 'ies'}**{location_phrase} in our portfolio. Here's what we have:\n\n",
            f"Here are the top options{location_phrase} from our curated collection:\n\n",
            f"I've matched **{len(props_to_show)} propert{'y' if len(props_to_show)==1 else 'ies'}**{location_phrase} that fits your search:\n\n",
        ]
        # Pick intro based on hash of query for variety
        intro = intro_options[hash(query) % len(intro_options)]
        response = intro

        for i, p in enumerate(props_to_show, 1):
            portal = p.get("developer_or_portal", "")
            badge = "🏰 DarGlobal" if portal == "DarGlobal" else "🏢 Wasalt"
            usd = p.get("price_usd", 0)
            usd_str = f" (~${usd:,})" if usd else ""
            features_list = (p.get("key_features") or [])[:4]

            response += (
                f"### {i}. {p.get('title')}\n"
                f"{badge} · 📍 **{p.get('district')}, {p.get('city')}, {p.get('country')}**\n\n"
                f"| Detail | Info |\n"
                f"|--------|------|\n"
                f"| 💰 Price | **{p.get('price')}**{usd_str} |\n"
                f"| 🏠 Type | {p.get('property_type')} |\n"
                f"| 🛏 Bedrooms | {p.get('bedrooms')} |\n"
                f"| 🚿 Bathrooms | {p.get('bathrooms')} |\n"
                f"| 📐 Area | {p.get('area_sqft')} |\n"
                f"| 📋 Status | {p.get('status')} |\n\n"
                f"**Overview:** {p.get('description', '')}\n\n"
            )
            if features_list:
                response += "**Key Highlights:**\n"
                for feat in features_list:
                    response += f"- {feat}\n"
            response += f"\n🔗 [View Full Listing]({p.get('source_url')})\n\n---\n\n"

        response += (
            "Would you like more details on any of these properties, "
            "want to compare them, or shall I search for a different type or location?"
        )
        return response

    def build_chat_context(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], str]:
        """
        Prepare matched properties + LLM messages without calling the AI.
        Used by the streaming endpoint in app.py.
        Returns: (matched_properties, messages, corrected_query)
        """
        corrected = _correct_query(user_message)
        matched_properties = self.search_properties(corrected, top_k=4, filters=filters)
        context_str = self._build_context_prompt(matched_properties)
        location_intent = self._detect_location_intent(corrected)

        location_note = ""
        if location_intent:
            aliases = LOCATION_ALIASES.get(location_intent, [])
            has_exact = any(
                any(a in f"{p.get('district','')} {p.get('city','')}".lower() for a in aliases)
                for p in matched_properties
            )
            if not has_exact:
                location_note = (
                    f"\n\nIMPORTANT: The user asked about '{location_intent.title()}' but our database "
                    f"does not currently have listings EXACTLY there. The properties above are the closest "
                    f"alternatives. Acknowledge this honestly and warmly, then present the alternatives "
                    f"with an explanation of why they might also appeal to the user."
                )

        system_instruction = (
            "You are Alex, a warm and genuinely passionate luxury real estate advisor who loves helping people find their perfect property. "
            "You represent two world-class portfolios:\n\n"
            "🏰 **DarGlobal** — A London Stock Exchange-listed developer crafting extraordinary branded residences with Pagani, Lamborghini, "
            "Missoni, Aston Martin, Elie Saab, Fendi Casa, W Hotels, and Trump — across Dubai, London, Muscat (Oman), Marbella (Spain), Jeddah, and Doha.\n\n"
            "🏢 **Wasalt** — Saudi Arabia's #1 REGA-licensed digital real estate platform with verified, trusted listings across Riyadh, Jeddah, and the broader KSA.\n\n"
            "YOUR PERSONALITY:\n"
            "- Warm, conversational, and genuinely enthusiastic — talk like a knowledgeable friend in the industry, not a sales robot\n"
            "- Use natural, human language: 'What I love about this one is...', 'Between you and me...', 'This is a smart pick because...'\n"
            "- Be direct and honest — if something isn't in the portfolio, say so warmly and pivot to something better\n\n"
            "HOW TO RESPOND:\n"
            "1. NEVER open with robotic phrases like 'Here are the top listings matching...' — start naturally\n"
            "2. Acknowledge what they asked for, then dive into properties with personality\n"
            "3. Present each property with life: what makes it special, the lifestyle it unlocks, why it's worth it\n"
            "4. Use clean markdown: ### for property names, **bold** for key facts, tables for quick spec comparisons\n"
            "5. Always mention price in both local currency AND USD\n"
            "6. Add real insight: investment angle, neighbourhood vibe, what sets it apart\n"
            "7. End with a warm, specific follow-up — not generic\n"
            "8. NEVER invent properties, prices, or facts not in the context below\n\n"
            f"--- AVAILABLE PROPERTIES ---\n{context_str}\n"
            f"----------------------------{location_note}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_instruction}]
        if history:
            for turn in history[-8:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})

        return matched_properties, messages, corrected

    # ------------------------------------------------------------------
    # Main chat
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline:
        1. Correct typos, detect intent
        2. Retrieve relevant properties
        3. Build system prompt with context
        4. Call OpenRouter (5 models with fallback)
        5. Return structured response
        """
        corrected = _correct_query(user_message)
        matched_properties = self.search_properties(corrected, top_k=4, filters=filters)
        context_str = self._build_context_prompt(matched_properties)
        location_intent = self._detect_location_intent(corrected)

        # Build location note for system prompt
        location_note = ""
        if location_intent:
            aliases = LOCATION_ALIASES.get(location_intent, [])
            has_exact = any(
                any(a in f"{p.get('district','')} {p.get('city','')}".lower() for a in aliases)
                for p in matched_properties
            )
            if not has_exact:
                location_note = (
                    f"\n\nIMPORTANT: The user asked about '{location_intent.title()}' but our database "
                    f"does not currently have listings EXACTLY there. The properties above are the closest "
                    f"alternatives. Acknowledge this honestly and warmly, then present the alternatives "
                    f"with an explanation of why they might also appeal to the user."
                )

        system_instruction = (
            "You are Alex, a warm and genuinely passionate luxury real estate advisor who loves helping people find their perfect property. "
            "You represent two world-class portfolios:\n\n"
            "🏰 **DarGlobal** — A London Stock Exchange-listed developer crafting extraordinary branded residences with Pagani, Lamborghini, "
            "Missoni, Aston Martin, Elie Saab, Fendi Casa, W Hotels, and Trump — across Dubai, London, Muscat (Oman), Marbella (Spain), Jeddah, and Doha.\n\n"
            "🏢 **Wasalt** — Saudi Arabia's #1 REGA-licensed digital real estate platform with verified, trusted listings across Riyadh, Jeddah, and the broader KSA.\n\n"
            "YOUR PERSONALITY:\n"
            "- Warm, conversational, and genuinely enthusiastic — talk like a knowledgeable friend in the industry, not a sales robot\n"
            "- Use natural, human language: 'What I love about this one is...', 'Between you and me...', 'This is a smart pick because...'\n"
            "- Be direct and honest — if something isn't in the portfolio, say so warmly and pivot to something better\n"
            "- Show real curiosity about what the user is looking for\n\n"
            "HOW TO RESPOND:\n"
            "1. NEVER open with robotic phrases like 'Here are the top listings matching...' — start naturally\n"
            "2. Acknowledge what they asked for first, then dive into the properties with personality\n"
            "3. Present each property with life: what makes it special, the lifestyle it unlocks, why it's worth considering\n"
            "4. Use clean markdown: ### for property names, **bold** for key facts, tables for quick spec comparisons\n"
            "5. Always mention price in both local currency AND USD — make it easy to compare\n"
            "6. Add real insight: investment angle, neighbourhood vibe, what sets it apart from alternatives\n"
            "7. If a location isn't in our portfolio, say something like: 'We don't have listings there right now, but let me show you something you might love even more...'\n"
            "8. End with a warm, specific follow-up — not generic. E.g. 'Would you like to compare this with the W Residences nearby?' or 'Are you looking for investment or a family home?'\n"
            "9. NEVER invent properties, prices, or facts not in the context below\n"
            "10. Keep responses conversational but informative — imagine chatting over coffee, not filing a report\n\n"
            f"--- AVAILABLE PROPERTIES ---\n{context_str}\n"
            f"----------------------------{location_note}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_instruction}]

        if history:
            for turn in history[-8:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})

        # --- Offline mode ---
        if not self.client:
            logger.info("No OpenRouter client — using conversational fallback.")
            return {
                "response": self._generate_conversational_fallback(user_message, matched_properties, corrected),
                "properties": matched_properties,
                "model": "offline-fallback",
                "status": "success",
            }

        # --- Try models ---
        models_to_try = [self.primary_model] + [m for m in DEFAULT_MODELS if m != self.primary_model]
        last_error: Optional[str] = None

        for model in models_to_try:
            try:
                logger.info(f"Calling OpenRouter model: {model}")
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.75,
                    max_tokens=1400,
                )
                answer = completion.choices[0].message.content or ""
                if answer.strip():
                    return {
                        "response": answer,
                        "properties": matched_properties,
                        "model": model,
                        "status": "success",
                    }
            except OpenAIError as exc:
                logger.warning(f"Model {model} OpenAI error: {exc}")
                last_error = str(exc)
            except Exception as exc:
                logger.warning(f"Model {model} unexpected error: {exc}")
                last_error = str(exc)

        logger.error(f"All OpenRouter models failed. Last: {last_error}")
        return {
            "response": self._generate_conversational_fallback(user_message, matched_properties, corrected),
            "properties": matched_properties,
            "model": "local-fallback",
            "error_detail": last_error,
            "status": "fallback",
        }
