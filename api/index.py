import os
import json
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Response, HTTPException
from notion_client import Client as NotionClient
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

app = FastAPI()

# --- ENV ---
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
NOTION_TASKS_DB_ID = os.environ.get("NOTION_TASKS_DB_ID", "")
NOTION_TEAM_DB_ID = os.environ.get("NOTION_TEAM_DB_ID", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BOT_ADMIN_PHONE = os.environ.get("BOT_ADMIN_PHONE", "")  # optional

notion = NotionClient(auth=NOTION_TOKEN)
oai = OpenAI(api_key=OPENAI_API_KEY)

# ---------- helpers ----------

def twiml(text: str) -> Response:
    r = MessagingResponse()
    r.message(text)
    return Response(content=str(r), media_type="application/xml")

def normalize_phone(p: str) -> str:
    # Twilio WhatsApp From is like "whatsapp:+14155552671"
    if p.startswith("whatsapp:"):
        p = p.replace("whatsapp:", "")
    return p.strip()

def get_full_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    path = request.url.path
    query = request.url.query
    base = f"{proto}://{host}{path}"
    return base + (f"?{query}" if query else "")

async def require_twilio_signature(request: Request, form: Dict[str, str]) -> None:
    sig = request.headers.get("x-twilio-signature", "")
    if not (TWILIO_AUTH_TOKEN and sig):
        return  # allow dev if missing, but you should set this in prod

    url = get_full_url(request)
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    if not validator.validate(url, form, sig):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

def iso_week_key(now: Optional[dt.date] = None) -> str:
    d = now or dt.date.today()
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"

def safe_text(prop: Any) -> str:
    try:
        return prop[0]["plain_text"]
    except Exception:
        return ""

# ---------- Notion access ----------

def notion_find_member_by_phone(phone_e164: str) -> Optional[Dict[str, str]]:
    # Team DB property must be a Phone field named "WhatsApp"
    resp = notion.databases.query(
        database_id=NOTION_TEAM_DB_ID,
        filter={
            "property": "WhatsApp",
            "phone_number": {"equals": phone_e164}
        }
    )
    if not resp.get("results"):
        return None
    page = resp["results"][0]
    name = safe_text(page["properties"]["Name"]["title"])
    return {"id": page["id"], "name": name or "Team member"}

def notion_get_tasks_for_member_week(member_id: str, week: str) -> List[Dict[str, Any]]:
    resp = notion.databases.query(
        database_id=NOTION_TASKS_DB_ID,
        filter={
            "and": [
                {"property": "Owner", "relation": {"contains": member_id}},
                {"property": "Week", "select": {"equals": week}},
            ]
        },
        sorts=[{"property": "Due Date", "direction": "ascending"}],
        page_size=50,
    )
    out = []
    for p in resp.get("results", []):
        props = p["properties"]
        out.append({
            "id": p["id"],
            "title": safe_text(props["Task"]["title"]) or "Untitled",
            "status": (props.get("Status", {}).get("select") or {}).get("name"),
            "progress": props.get("Progress %", {}).get("number"),
            "due": (props.get("Due Date", {}).get("date") or {}).get("start"),
        })
    return out

def notion_update_task(task_id: str, status: Optional[str], progress: Optional[int], blocker: Optional[str], needs_attention: Optional[bool]) -> None:
    properties: Dict[str, Any] = {
        "Last Update": {"date": {"start": dt.datetime.utcnow().isoformat()}}
    }
    if status:
        properties["Status"] = {"select": {"name": status}}
    if progress is not None:
        properties["Progress %"] = {"number": int(progress)}
    if blocker is not None:
        properties["Blocker"] = {"rich_text": [{"text": {"content": blocker[:1800]}}]}
    if needs_attention is not None:
        properties["Needs Attention"] = {"checkbox": bool(needs_attention)}

    notion.pages.update(page_id=task_id, properties=properties)

def notion_append_log(task_id: str, line: str) -> None:
    page = notion.pages.retrieve(page_id=task_id)
    current = ""
    try:
        current = "".join([r.get("plain_text", "") for r in page["properties"]["Update Log"]["rich_text"]])
    except Exception:
        current = ""
    next_text = (current + "\n" + line).strip()
    notion.pages.update(
        page_id=task_id,
        properties={"Update Log": {"rich_text": [{"text": {"content": next_text[:1800]}}]}}
    )

# ---------- AI parsing ----------

def ai_parse_update(message: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns:
      {
        "task_index": int | null,
        "task_id": str | null,
        "status": str | null,
        "progress": int | null,
        "blocker": str | null,
        "needs_attention": bool | null,
        "confidence": float,
        "followup": str | null
      }
    """
    task_list = []
    for i, t in enumerate(tasks):
        letter = chr(65 + i)
        task_list.append({
            "letter": letter,
            "id": t["id"],
            "title": t["title"],
            "status": t.get("status"),
            "progress": t.get("progress"),
            "due": t.get("due"),
        })

    prompt = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "You are a task update parser for a WhatsApp bot.\n"
                    "Given the user's message and their task list, return STRICT JSON only.\n\n"
                    "Rules:\n"
                    "- Choose exactly one task, or ask a follow-up if unclear.\n"
                    "- If user says done/completed, set status=Done and progress=100.\n"
                    "- If user says blocked, set status=Blocked and blocker text.\n"
                    "- If user gives a percent, set progress.\n"
                    "- needs_attention=true if blocked or overdue or user signals risk.\n"
                    "- confidence 0 to 1.\n\n"
                    f"Tasks: {json.dumps(task_list)}\n\n"
                    f"User message: {message}\n\n"
                    "Return JSON keys: task_id, status, progress, blocker, needs_attention, confidence, followup\n"
                )
            }
        ],
    }

    resp = oai.responses.create(
        model="gpt-5.2",
        input=[prompt],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.output_text)

    # Basic sanity
    data.setdefault("task_id", None)
    data.setdefault("status", None)
    data.setdefault("progress", None)
    data.setdefault("blocker", None)
    data.setdefault("needs_attention", None)
    data.setdefault("confidence", 0.0)
    data.setdefault("followup", None)
    return data

def build_weekly_message(week: str, tasks: List[Dict[str, Any]]) -> str:
    lines = [f"Your tasks for {week}:"]
    for i, t in enumerate(tasks):
        letter = chr(65 + i)
        due = f" (Due {t['due']})" if t.get("due") else ""
        lines.append(f"{letter}) {t['title']}{due}")
    lines += [
        "",
        "Reply like:",
        "A done",
        "B 60%",
        "C blocked waiting on X",
        "Or just type normally, I’ll parse it."
    ]
    return "\n".join(lines)

# ---------- Routes ----------

@app.post("/api/twilio/inbound")
async def twilio_inbound(request: Request):
    form = await request.form()
    form_dict = {k: str(v) for k, v in form.items()}

    await require_twilio_signature(request, form_dict)

    from_raw = form_dict.get("From", "")
    body = (form_dict.get("Body") or "").strip()

    from_phone = normalize_phone(from_raw)
    if not body:
        return twiml("Send an update like: 'done', '60%', or 'blocked waiting on X'.")

    member = notion_find_member_by_phone(from_phone)
    if not member:
        return twiml("I don’t recognize this number yet. Add it to the Team database in Notion under WhatsApp.")

    week = iso_week_key()
    tasks = notion_get_tasks_for_member_week(member["id"], week)
    if not tasks:
        return twiml(f"No tasks found for you in {week}. Ask your manager to assign tasks in Notion.")

    parsed = ai_parse_update(body, tasks)

    confidence = float(parsed.get("confidence") or 0.0)
    followup = parsed.get("followup")

    if confidence < 0.7 or followup:
        q = followup or "Which task is this for? Reply with A, B, C, or paste the task name."
        return twiml(q)

    task_id = parsed.get("task_id")
    if not task_id:
        return twiml("I couldn’t match that to a task. Reply with A, B, C, or paste the task name.")

    status = parsed.get("status")
    progress = parsed.get("progress")
    blocker = parsed.get("blocker")
    needs_attention = parsed.get("needs_attention")

    # Update Notion
    notion_update_task(task_id, status, progress, blocker, needs_attention)
    notion_append_log(task_id, f"{dt.datetime.utcnow().isoformat()} | {member['name']}: {body}")

    # Reply tone
    if status == "Done":
        return twiml("Nice. Marked as Done and updated Notion.")
    if status == "Blocked":
        return twiml("Got it. Marked Blocked in Notion. What do you need to unblock?")
    return twiml("Got it. Updated Notion.")

@app.get("/api/health")
async def health():
    return {"ok": True, "week": iso_week_key()}

@app.post("/api/jobs/send_weekly")
async def send_weekly(request: Request):
    # Simple shared secret so only you can trigger jobs
    secret = request.headers.get("x-bot-secret", "")
    if secret != os.environ.get("BOT_SECRET", ""):
        raise HTTPException(status_code=403, detail="Forbidden")

    payload = await request.json()
    member_phone = payload.get("to")  # +E164
    week = payload.get("week") or iso_week_key()

    member = notion_find_member_by_phone(member_phone)
    if not member:
        return {"ok": False, "error": "Member not found in Notion Team DB"}

    tasks = notion_get_tasks_for_member_week(member["id"], week)
    msg = build_weekly_message(week, tasks)

    # Twilio send is usually done via Twilio REST API.
    # For Vercel simplicity, you can trigger weekly by using Twilio Studio Flow,
    # or add a small sender function later.
    return {"ok": True, "preview_message": msg, "tasks": len(tasks)}
