"""Create the authenticated moderation schema.

Revision ID: 20260711_0001
Revises: None
"""

from alembic import op
import sqlalchemy as sa


revision = "20260711_0001"
down_revision = None
branch_labels = None
depends_on = None


def _columns(inspector, table):
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "users" not in tables:
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("username", sa.String(64), nullable=False),
            sa.Column("password_hash", sa.String(255), nullable=False),
            sa.Column("role", sa.String(16), nullable=False, server_default="user"),
            sa.Column("warning_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("is_banned", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("username"),
        )
        op.create_index("ix_users_username", "users", ["username"], unique=True)
        op.create_index("ix_users_role", "users", ["role"])
    else:
        columns = _columns(inspector, "users")
        additions = {
            "password_hash": sa.Column("password_hash", sa.String(255), nullable=True),
            "role": sa.Column("role", sa.String(16), nullable=False, server_default="user"),
            "created_at": sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            "updated_at": sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        }
        with op.batch_alter_table("users") as batch:
            for name, column in additions.items():
                if name not in columns:
                    batch.add_column(column)
        op.execute("UPDATE users SET password_hash='!reset-required' WHERE password_hash IS NULL")

    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "message_logs" not in tables:
        op.create_table(
            "message_logs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("original_text", sa.Text(), nullable=True),
            sa.Column("encrypted_text", sa.Text(), nullable=True),
            sa.Column("content_hash", sa.String(64), nullable=False),
            sa.Column("displayed_text", sa.Text(), nullable=False),
            sa.Column("is_toxic", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("toxicity_score", sa.Float(), nullable=False, server_default="0"),
            sa.Column("category", sa.String(32), nullable=False, server_default="clean"),
            sa.Column("severity", sa.String(16), nullable=False, server_default="none"),
            sa.Column("detection_method", sa.String(64), nullable=False, server_default="unknown"),
            sa.Column("action_taken", sa.String(32), nullable=False, server_default="passed"),
            sa.Column("review_status", sa.String(16), nullable=False, server_default="not_required"),
            sa.Column("decision_reason", sa.Text(), nullable=True),
            sa.Column("model_version", sa.String(128), nullable=False, server_default="rules-only"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        )
    else:
        columns = _columns(inspector, "message_logs")
        additions = {
            "encrypted_text": sa.Column("encrypted_text", sa.Text(), nullable=True),
            "content_hash": sa.Column("content_hash", sa.String(64), nullable=True),
            "displayed_text": sa.Column("displayed_text", sa.Text(), nullable=True),
            "category": sa.Column("category", sa.String(32), server_default="clean"),
            "severity": sa.Column("severity", sa.String(16), server_default="none"),
            "detection_method": sa.Column("detection_method", sa.String(64), server_default="legacy"),
            "action_taken": sa.Column("action_taken", sa.String(32), server_default="passed"),
            "review_status": sa.Column("review_status", sa.String(16), server_default="not_required"),
            "decision_reason": sa.Column("decision_reason", sa.Text(), nullable=True),
            "model_version": sa.Column("model_version", sa.String(128), server_default="legacy"),
            "expires_at": sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        }
        with op.batch_alter_table("message_logs") as batch:
            for name, column in additions.items():
                if name not in columns:
                    batch.add_column(column)
        if "censored_text" in columns:
            op.execute("UPDATE message_logs SET displayed_text=censored_text WHERE displayed_text IS NULL")
        op.execute("UPDATE message_logs SET content_hash='legacy-' || id WHERE content_hash IS NULL")

    if "appeals" not in set(sa.inspect(bind).get_table_names()):
        op.create_table(
            "appeals",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("message_id", sa.Integer(), sa.ForeignKey("message_logs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("reason", sa.Text(), nullable=False),
            sa.Column("status", sa.String(16), nullable=False, server_default="open"),
            sa.Column("admin_note", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade():
    # Legacy veri kaybını önlemek için otomatik downgrade bilinçli olarak yoktur.
    raise RuntimeError("Bu migration veri kaybı riski nedeniyle otomatik geri alınamaz.")
