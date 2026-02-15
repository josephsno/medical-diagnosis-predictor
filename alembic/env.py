import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, pool
from alembic import context

load_dotenv()

from database.base_model import Base
from apps.cases.models import Case
from apps.symptoms.models import Symptom

# MetaData for autogenerate
target_metadata = Base.metadata

# use env variable instead of alembic.ini
DATABASE_URL = os.getenv("DATABASE_URL")
config = context.config  # Alembic config object

# -------------------------------
# Offline migrations (sql script only)
# -------------------------------
def run_migrations_offline():
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
# Online migrations (applied to DB)
# -------------------------------
def run_migrations_online():
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


# -------------------------------
# Run
# -------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
