import os
from logging.config import fileConfig
from sqlalchemy import create_engine, pool
from alembic import context
from dotenv import dotenv_values

# -------------------------------
# Load .env directly, bypassing shell cache
# -------------------------------
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
config_values = dotenv_values(dotenv_path)
DATABASE_URL = config_values.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(f"DATABASE_URL not found in {dotenv_path}")

# Alembic Config object
config = context.config

# Override sqlalchemy.url from .env
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# Logging config
fileConfig(config.config_file_name)

# Import your Base for autogenerate
from database.base_model import Base
from apps.cases.models import Case
from apps.symptoms.models import Symptom

target_metadata = Base.metadata

# -------------------------------
# Offline migrations
# -------------------------------
def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

# -------------------------------
# Online migrations
# -------------------------------
def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

# -------------------------------
# Run migrations based on mode
# -------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
