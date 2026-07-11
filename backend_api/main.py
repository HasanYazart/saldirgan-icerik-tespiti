"""FastAPI uygulama giriş noktası."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import models, schemas
from .censor_service import censor_message
from .crypto import content_fingerprint, decrypt_text, encrypt_text
from .database import SessionLocal, engine, get_db
from .ml_service import ml_service
from .rate_limit import build_rate_limiter
from .security import (
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)
from .settings import settings


STATIC_DIR = Path(__file__).resolve().parent / "static"
rate_limiter = build_rate_limiter(
    settings.redis_url, settings.moderation_rate_limit, settings.rate_limit_window_seconds
)


def _bootstrap_admin(db: Session) -> None:
    if not settings.admin_username:
        return
    username = settings.admin_username.lower()
    user = db.scalar(select(models.User).where(models.User.username == username))
    if user is None:
        db.add(
            models.User(
                username=username,
                password_hash=hash_password(settings.admin_password),
                role="admin",
            )
        )
        db.commit()
    elif user.role != "admin":
        user.role = "admin"
        db.commit()


def _purge_expired_records(db: Session) -> None:
    now = datetime.now(timezone.utc)
    expired = db.scalars(
        select(models.MessageLog).where(
            models.MessageLog.expires_at.is_not(None), models.MessageLog.expires_at < now
        )
    ).all()
    for record in expired:
        db.delete(record)
    if expired:
        db.commit()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    models.Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        _bootstrap_admin(db)
        _purge_expired_records(db)
    ml_service.ensure_available()
    yield


app = FastAPI(
    title=settings.app_name,
    version="3.0.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
    )
    return response


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/auth/register", response_model=schemas.TokenResponse, status_code=201)
def register(payload: schemas.RegisterRequest, db: Session = Depends(get_db)):
    if not settings.allow_registration:
        raise HTTPException(status_code=403, detail="Yeni kayıtlar kapalı.")
    user = models.User(
        username=payload.username.lower(),
        password_hash=hash_password(payload.password),
        role="user",
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Bu kullanıcı adı zaten kullanılıyor.")
    return schemas.TokenResponse(access_token=create_access_token(user))


@app.post("/api/auth/login", response_model=schemas.TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(models.User).where(models.User.username == form.username.lower()))
    if user is None or not verify_password(form.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı adı veya parola hatalı.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return schemas.TokenResponse(access_token=create_access_token(user))


@app.get("/api/me/status", response_model=schemas.UserStatusResponse)
def current_status(current_user: models.User = Depends(get_current_user)):
    return schemas.UserStatusResponse(
        user_id=current_user.id,
        username=current_user.username,
        role=current_user.role,
        warning_count=current_user.warning_count,
        is_banned=current_user.is_banned,
    )


@app.post("/api/chat/send", response_model=schemas.MessageResponse)
def send_message(
    message: schemas.MessageCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    rate_limiter.check(str(current_user.id))
    if current_user.is_banned:
        raise HTTPException(status_code=403, detail="Hesap moderasyon nedeniyle kısıtlandı.")

    analysis = ml_service.analyze_text(message.text, user_id=current_user.id)
    is_toxic = bool(analysis["is_toxic"])
    needs_review = bool(analysis["needs_review"])
    displayed_text = message.text
    action_taken = "passed"
    review_status = "not_required"
    status_value = "clean"

    if is_toxic:
        displayed_text = censor_message(message.text)
        current_user.warning_count += 1
        action_taken = "warned"
        status_value = "toxic"
        if (
            settings.auto_ban_warning_count > 0
            and current_user.warning_count >= settings.auto_ban_warning_count
        ):
            current_user.is_banned = True
            action_taken = "banned"
    elif needs_review:
        displayed_text = "[Bu mesaj insan incelemesi bekliyor]"
        action_taken = "queued_for_review"
        review_status = "pending"
        status_value = "review"

    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.message_retention_days)
    log = models.MessageLog(
        user_id=current_user.id,
        original_text=message.text if settings.store_raw_messages else None,
        encrypted_text=encrypt_text(message.text),
        content_hash=content_fingerprint(message.text),
        displayed_text=displayed_text,
        is_toxic=is_toxic,
        toxicity_score=analysis["toxicity_score"],
        category=analysis["category"],
        severity=analysis["severity"],
        detection_method=analysis["detection_method"],
        action_taken=action_taken,
        review_status=review_status,
        decision_reason=f"threshold={ml_service.toxicity_threshold}",
        model_version=analysis["model_version"],
        expires_at=expires_at,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return schemas.MessageResponse(
        message_id=log.id,
        status=status_value,
        displayed_text=displayed_text,
        warning_count=current_user.warning_count,
        action_taken=action_taken,
        detection_method=analysis["detection_method"],
        toxicity_score=analysis["toxicity_score"],
        category=analysis["category"],
        severity=analysis["severity"],
        model_version=analysis["model_version"],
    )


@app.post("/api/appeals", response_model=schemas.AppealResponse, status_code=201)
def create_appeal(
    payload: schemas.AppealCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    message = db.get(models.MessageLog, payload.message_id)
    if message is None or message.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Moderasyon kaydı bulunamadı.")
    if message.action_taken == "passed":
        raise HTTPException(status_code=400, detail="Geçen mesaj için itiraz oluşturulamaz.")
    existing = db.scalar(
        select(models.Appeal).where(
            models.Appeal.message_id == message.id, models.Appeal.status == "open"
        )
    )
    if existing:
        raise HTTPException(status_code=409, detail="Bu kayıt için açık bir itiraz zaten var.")
    appeal = models.Appeal(message_id=message.id, user_id=current_user.id, reason=payload.reason)
    db.add(appeal)
    db.commit()
    db.refresh(appeal)
    return appeal


@app.get("/api/admin/reviews", response_model=list[schemas.AdminModerationRecord])
def list_reviews(
    review_status: str = Query("pending", pattern=r"^(pending|not_required|confirmed|overturned)$"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    records = db.scalars(
        select(models.MessageLog)
        .where(models.MessageLog.review_status == review_status)
        .order_by(models.MessageLog.created_at.desc())
        .limit(limit)
    ).all()
    return [
        schemas.AdminModerationRecord.model_validate(record).model_copy(
            update={"content": decrypt_text(record.encrypted_text)}
        )
        for record in records
    ]


@app.post("/api/admin/reviews/{message_id}", response_model=schemas.ModerationRecord)
def decide_review(
    message_id: int,
    payload: schemas.ReviewDecision,
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    record = db.get(models.MessageLog, message_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Moderasyon kaydı bulunamadı.")
    user = db.get(models.User, record.user_id)
    if user is None:
        raise HTTPException(status_code=409, detail="Kayda bağlı kullanıcı bulunamadı.")
    if payload.decision == "approved":
        record.review_status = "confirmed"
        if not record.is_toxic:
            record.is_toxic = True
            record.action_taken = "warned"
            user.warning_count += 1
    else:
        record.review_status = "overturned"
        if record.is_toxic and user.warning_count > 0:
            user.warning_count -= 1
        record.is_toxic = False
        record.action_taken = "passed"
        user.is_banned = False
    record.decision_reason = payload.admin_note
    db.commit()
    db.refresh(record)
    return record


@app.get("/api/admin/appeals", response_model=list[schemas.AppealResponse])
def list_appeals(
    db: Session = Depends(get_db), _admin: models.User = Depends(require_admin)
):
    return db.scalars(
        select(models.Appeal).where(models.Appeal.status == "open").order_by(models.Appeal.created_at)
    ).all()


@app.post("/api/admin/appeals/{appeal_id}", response_model=schemas.AppealResponse)
def decide_appeal(
    appeal_id: int,
    payload: schemas.AppealDecision,
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    appeal = db.get(models.Appeal, appeal_id)
    if appeal is None or appeal.status != "open":
        raise HTTPException(status_code=404, detail="Açık itiraz bulunamadı.")
    appeal.status = payload.decision
    appeal.admin_note = payload.admin_note
    appeal.resolved_at = datetime.now(timezone.utc)
    if payload.decision == "accepted":
        record = appeal.message
        user = appeal.user
        if record.is_toxic and user.warning_count > 0:
            user.warning_count -= 1
        user.is_banned = False
        record.is_toxic = False
        record.action_taken = "passed"
        record.review_status = "overturned"
        record.decision_reason = payload.admin_note
    db.commit()
    db.refresh(appeal)
    return appeal


@app.post("/api/admin/users/{user_id}/reset")
def reset_user(
    user_id: int,
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_admin),
):
    user = db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")
    user.warning_count = 0
    user.is_banned = False
    db.commit()
    return {"status": "ok"}


@app.get("/api/health")
def health_check():
    return {"status": "online" if not ml_service.is_mock else "degraded", **ml_service.health()}


@app.get("/api/ready")
def readiness(response: Response):
    health = ml_service.health()
    if ml_service.is_mock and not settings.allow_mock_model:
        response.status_code = 503
    return health
