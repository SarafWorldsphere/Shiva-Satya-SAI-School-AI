from datetime import datetime, timezone
from sqlalchemy import BigInteger, Column, DateTime, Integer, String
from config.database import Base


class AIUsageLog(Base):
    __tablename__ = "ai_usage_logs"

    ai_usage_id = Column(BigInteger, primary_key=True, index=True)

    # 1. Client & User Identification
    client_name = Column(String, index=True)  # e.g., 'SSS', 'SGS', 'Demo'
    user_email = Column(String, index=True)   # Email of the Student/Faculty/HM/Admin/Parent

    # 2. Module & Feature Tracking (matches public.ai_usage_logs.feature_used)
    module_name = Column(String, index=True)  # e.g., 'Student Dashboard', 'Faculty Dashboard'
    feature_used = Column(String)             # e.g., 'Quiz Generator', 'Language Translator'

    # 3. Token Tracking
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)

    # 4. Timestamp (matches public.ai_usage_logs.created_at)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))