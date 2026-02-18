import os, json, datetime as dt
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator

app = FastAPI()

TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

def twiml(msg: str) -> Response:
    r = MessagingResponse()
    r.message(msg)
    return Response(str(r), media_type="application/xml")

@app.get("/health")
async def health():
    return {"ok": True, "ts": dt.datetime.utcnow().isoformat()}

@app.post("/twilio/inbound")
async def inbound(request: Request):
    form = await request.form()
    form = {k: str(v) for k, v in form.items()}

    # Verify Twilio signature in prod
    sig = request.headers.get("x-twilio-signature", "")
    if TWILIO_AUTH_TOKEN and sig:
        url = str(request.url)
        validator = RequestValidator(TWILIO_AUTH_TOKEN)
        if not validator.validate(url, form, sig):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    body = (form.get("Body") or "").strip()
    if not body:
        return twiml("Send an update like: done, 60%, or blocked waiting on X.")

    # Replace this with Notion + AI logic next
    return twiml(f"Got it: {body}")
