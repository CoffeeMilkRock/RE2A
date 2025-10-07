# rasa-bot/actions/actions.py
from __future__ import annotations
from typing import Any, Dict, List, Text


from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import os
import re, time, requests
from .utils import get_persona_defaults, natural_search
API_BASE = os.getenv("REA_API_BASE", "http://localhost:8008/api/v2")

VALID_TYPES = {"căn hộ","nhà ở","nhà mặt tiền","studio","nhà vùng ven"}

class ActionAIRephrase(Action):
    def name(self) -> Text:
        return "action_ai_rephrase"


    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        last_user_msg = (tracker.latest_message.get("text") or "").strip()
        if not last_user_msg:
            dispatcher.utter_message(text="I didn't catch that. Could you please rephrase?")
            return []

        persona_name = tracker.get_slot("persona_name") or "Agent"
        persona_style = tracker.get_slot("persona_style") or "Helpful"
        persona_objective = tracker.get_slot("persona_objective") or "Assist with properties"

        prompt = (f"As {persona_name}, who is {persona_style} and aims to {persona_objective}, "
                  f"please rephrase the following message to be more engaging and clear:\n\n\"{last_user_msg}\"")

        # Call OpenAI API (pseudo-code, replace with actual API call)
        try:
            import google.generativeai as genai
            import os

            # Cấu hình API key từ biến môi trường
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            # Tạo model Gemini (chọn model bạn muốn, ở đây dùng gemini-1.5-flash để tiết kiệm chi phí)
            model = genai.GenerativeModel("gemini-2.5-flash")

            # Tạo prompt hội thoại
            prompt_text = f"You are a helpful real estate assistant.\nUser: {prompt}"

            # Gọi model sinh phản hồi
            response = model.generate_content(prompt_text)

            # Trích xuất text kết quả
            rephrased_text = response.text.strip() if response and response.text else "Sorry, I couldn't rephrase that."

            dispatcher.utter_message(text=rephrased_text)

        except Exception as e:
            dispatcher.utter_message(text=f"Sorry, I couldn't process that right now. {e}")

        return []

class ValidateSellPropertyForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_sell_property_form"

    async def validate_property_id(self, value, dispatcher, tracker, domain):
        if value and isinstance(value, str) and len(value.strip()) >= 3:
            return {"property_id": value.strip()}
        dispatcher.utter_message(text="Cung cấp id căn hộ giúp em nha (e.g., APT-D7-001).")
        return {"property_id": None}

    async def validate_unitId(self, value, dispatcher, tracker, domain):
        if value and isinstance(value, str) and len(value.strip()) >= 1:
            return {"unitId": value.strip()}
        dispatcher.utter_message(text="Unit id (e.g., U001).")
        return {"unitId": None}

    async def validate_location(self, value, dispatcher, tracker, domain):
        if value and isinstance(value, str) and len(value.strip()) >= 2:
            return {"location": value.strip()}
        dispatcher.utter_message(text="Anh muốn bán căn hộ ở đâu? (e.g., Quận 7)")
        return {"location": None}

    async def validate_property_type(self, value, dispatcher, tracker, domain):
        v = str(value).lower().strip() if value else ""
        if v in VALID_TYPES:
            return {"property_type": v}
        dispatcher.utter_message(text=f"Loại căn hộ là?: {', '.join(sorted(VALID_TYPES))}.")
        return {"property_type": None}

    async def validate_bedrooms(self, value, dispatcher, tracker, domain):
        try:
            v = float(value)
            if 0 < v < 10:
                return {"bedrooms": v}
        except: pass
        # try extract from text
        print(v)
        text = (tracker.latest_message.get("text") or "")
        print(text)
        m = re.search(r"(\d+)", text)
        if m:   
            return {"bedrooms": float(m.group(1))}
        dispatcher.utter_message(text="Có bao nhiêu phòng ngủ? (e.g., 1, 2, 3)")
        return {"bedrooms": None}

    async def validate_price(self, value, dispatcher, tracker, domain):
        try:
            v = float(value)
            if v > 0:
                return {"price": v}
        except: pass
        text = (tracker.latest_message.get("text") or "").replace(",", "")
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return {"price": float(m.group(1))}
        dispatcher.utter_message(text="Anh muốn bán bao nhiêu ạ? (e.g., 1000000000)")
        return {"price": None}

    async def validate_description(self, value, dispatcher, tracker, domain):
        if value and isinstance(value, str) and len(value.strip()) >= 5:
            return {"description": value.strip()}
        dispatcher.utter_message(text="Anh mô tả căn hộ ngắn gọn giúp em nha (Vị trí, không gian, nội thất,...).")
        return {"description": None}

# ---------- Embed Action ----------
class ActionEmbedProperty(Action):
    def name(self) -> Text:
        return "action_embed_property"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        # Gather slots
        pid = tracker.get_slot("property_id")
        unit = tracker.get_slot("unitId")
        loc = tracker.get_slot("location")
        ptype = tracker.get_slot("property_type")
        br = tracker.get_slot("bedrooms")
        price = tracker.get_slot("price")
        desc = tracker.get_slot("description")

        # Build payload matching your API schema (PropertyIn)
        payload = {
            "id": pid,
            "unitId": unit,
            "description": desc or "",
            "design_and_layout": {
                "location": loc, "type": ptype, "bedrooms": br, "price": price
            },
            "physical_features": {},
            "living_experience": {},
            "equipment_and_handover_materials": {},
            "legal_and_product_status": {},
            "property_groups": [],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            r = requests.post(f"{API_BASE}/property/embedding", json=payload, timeout=15)
            r.raise_for_status()
            # Expecting your standardized response { message, unitId, success, additional? }
            data = r.json()
            if data.get("success") is True:
                dispatcher.utter_message(response="utter_embed_success")
            else:
                dispatcher.utter_message(text=f"Embed response: {data}")
                dispatcher.utter_message(response="utter_embed_fail")
        except Exception as e:
            dispatcher.utter_message(text=f"Embed failed: {e}")
            dispatcher.utter_message(response="utter_embed_fail")

        # Optionally clear slots after saving:
        return [
            SlotSet("property_id", None),
            SlotSet("unitId", None),
            SlotSet("location", None),
            SlotSet("property_type", None),
            SlotSet("bedrooms", None),
            SlotSet("price", None),
            SlotSet("description", None),
        ]
class ActionInitializeConversation(Action):
    def name(self) -> Text:
        return "action_initialize_conversation"


    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        sender_id = tracker.sender_id
        persona = get_persona_defaults()
        return [
        SlotSet("conversation_id", sender_id),
        SlotSet("persona_name", persona.get("name", "Agent")),
        SlotSet("persona_style", persona.get("style", "Helpful")),
        SlotSet("persona_objective", persona.get("objective", "Assist with properties")),
        ]


class ValidatePropertySearchForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_property_search_form"


    async def extract_budget(self, dispatcher, tracker, domain):
    # Try to parse a number from the last user message
        import re
        text = (tracker.latest_message.get("text") or "").replace(",", "")
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return {"budget": float(m.group(1))}
        return {}


    async def validate_budget(self, value, dispatcher, tracker, domain):
        try:
            v = float(value)
            if v <= 0: raise ValueError
            return {"budget": v}
        except Exception:
            dispatcher.utter_message(text="Could you share a valid budget (e.g., 900)?")
        return {"budget": None}


class ActionSearchProperties(Action):
    def name(self) -> Text:
        return "action_search_properties"


    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        query = tracker.latest_message.get("text") or ""
        filters = {
        "location": tracker.get_slot("location"),
        "property_type": tracker.get_slot("property_type"),
        "budget_max": tracker.get_slot("budget"),
        "bedrooms": tracker.get_slot("bedrooms"),
        }
        # clean empty values
        filters = {k: v for k, v in filters.items() if v not in (None, "")}


        try:
            results = natural_search(query=query, filters=filters, top_k=5)
        except Exception as e:
            dispatcher.utter_message(text=f"I had trouble searching right now. {e}")
            return []

        items = results.get("items", [])
        if not items:
            dispatcher.utter_message(response="utter_no_results")
            return []


        # Render top results
        lines = []
        for it in items:
            meta = it.get("metadata", {})
            title = meta.get("title") or f"{meta.get('property_type','Property')} in {meta.get('location','?')}"
            price = meta.get("price")
            br = meta.get("bedrooms")
            desc = (meta.get("description") or "").strip()
            desc = (desc[:180] + "…") if len(desc) > 180 else desc
            lines.append(f"• {title} — {br} BR — ${price}\n {desc}")


          # Rephrase last user message
        print("Rephrasing user message...".join(lines))
        dispatcher.utter_message(text="Here are some properties I found:\n" + "\n".join(lines))
        return []