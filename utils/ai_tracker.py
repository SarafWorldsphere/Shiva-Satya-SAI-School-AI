from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from models import AIUsageLog

_TENANT_CACHE: dict[str, str] = {}


def _resolve_tenant_id(db: Session, client_name: str, user_email: str) -> Optional[str]:
    cache_key = (client_name or "").strip().lower()
    if cache_key and cache_key in _TENANT_CACHE:
        return _TENANT_CACHE[cache_key]

    row = None
    if client_name:
        row = db.execute(
            text(
                """
                SELECT tenant_id::text
                FROM ai_usage_logs
                WHERE lower(client_name) = lower(:cname) AND tenant_id IS NOT NULL
                LIMIT 1
                """
            ),
            {"cname": client_name.strip()},
        ).fetchone()

    if not row and user_email:
        row = db.execute(
            text(
                """
                SELECT tenant_id::text
                FROM ai_usage_logs
                WHERE lower(user_email) = lower(:uemail) AND tenant_id IS NOT NULL
                LIMIT 1
                """
            ),
            {"uemail": user_email.strip()},
        ).fetchone()

    if not row:
        row = db.execute(
            text("SELECT tenant_id::text FROM platform.tenants LIMIT 1")
        ).fetchone()

    tid = row[0] if row else None
    if tid and cache_key:
        _TENANT_CACHE[cache_key] = tid
    return tid


def log_gemini_usage(
    db: Session,
    response,
    client_name: str,
    user_email: str,
    module_name: str,
    feature_name: str,
    tenant_id: Optional[str] = None,
):
    """
    Extracts token usage from a Gemini AI response and logs it to public.ai_usage_logs.
    """
    prompt_tokens = 0
    completion_tokens = 0

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    total_tokens = prompt_tokens + completion_tokens

    try:
        resolved_tid = tenant_id or _resolve_tenant_id(db, client_name, user_email)
        if resolved_tid:
            db.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(resolved_tid)},
            )

        new_log = AIUsageLog(
            client_name=client_name or "SSS",
            user_email=user_email or "",
            module_name=module_name,
            feature_used=feature_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            created_at=datetime.now(timezone.utc),
        )

        db.add(new_log)
        db.commit()
        db.refresh(new_log)
    except Exception as e:
        print(f"⚠️ Failed to log AI usage to database: {e}")
        db.rollback()