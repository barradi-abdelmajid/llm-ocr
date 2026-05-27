"""LM Studio client — health check and gap analysis."""

from config import LM_STUDIO_HOST, LM_VERIFY
import httpx


def check_lm_studio():
    try:
        r = httpx.post(
            f"{LM_STUDIO_HOST}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
            timeout=10,
        )
        body = r.json()
        loaded = body.get("model", "unknown") if r.status_code == 200 else "unknown"
        print(f"  LM Studio reachable (loaded: {loaded})")
        return True
    except Exception as e:
        print(f"  LM Studio not reachable ({e})")
        return False


def llm_analyze_report(report_text):
    prompt = (
        "You analyze a PDF page segmentation report of handwritten EE notes. "
        "Identify suspicious gaps or issues. Reply with specific, actionable suggestions.\n\n"
        f"{report_text[:4000]}"
    )
    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
            r = client.post(
                f"{LM_STUDIO_HOST}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0,
                },
            )
        body = r.json()
        if "choices" in body and len(body["choices"]) > 0:
            return body["choices"][0]["message"]["content"].strip()
        return f"Unexpected response: {str(body)[:500]}"
    except Exception as e:
        return f"ERROR: {e}"
