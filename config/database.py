import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Pull database credentials from your secure .env file
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "").strip()

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    raise ValueError("⚠️ Database credentials are missing from the .env file!")

# Construct the PostgreSQL connection string
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create the database engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# If DEFAULT_TENANT_ID is configured in .env, apply it per checkout so stamp_tenant_id() / RLS sees it
if DEFAULT_TENANT_ID:
    @event.listens_for(engine, "checkout")
    def _set_default_tenant_on_checkout(dbapi_conn, connection_record, connection_proxy):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("SELECT set_config('app.tenant_id', %s, false)", (DEFAULT_TENANT_ID,))
        finally:
            cursor.close()

# Create a customized session class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for your database models (schemas) to inherit from
Base = declarative_base()

# Import models AFTER Base is defined so metadata is registered without circular import
from models import AIUsageLog  # noqa: E402, F401


# Dependency function to use in your routes to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


print("✅ Database configuration loaded successfully.")