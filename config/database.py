from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.orm import Session
from models import AIUsageLog


def log_gemini_usage(
    db: Session,
    response,
    client_name: str,
    user_email: str,
    module_name: str,
    feature_name: str,
    tenant_id: str | None = None,
):
    prompt_tokens = 0
    completion_tokens = 0

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    total_tokens = prompt_tokens + completion_tokens

    try:
        if tenant_id:
            db.execute(text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)})

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