from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, schemas
from .database import SessionLocal, engine, get_db
from .ml_service import ml_service
from .censor_service import censor_message

# Veritabanı tablolarını oluştur
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Saldırgan İçerik Tespiti API", version="1.0")

@app.post("/api/chat/send", response_model=schemas.MessageResponse)
def send_message(message: schemas.MessageCreate, db: Session = Depends(get_db)):
    # 1. Kullanıcıyı bul veya oluştur
    user = db.query(models.User).filter(models.User.id == message.user_id).first()
    if not user:
        user = models.User(id=message.user_id, username=message.username)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Ban kontrolü
    if user.is_banned:
        raise HTTPException(status_code=403, detail="Kullanıcı banlanmış durumda mesaj gönderemez.")

    # 2. Yapay Zeka ile Mesajı Analiz Et
    analysis = ml_service.analyze_text(message.text)
    is_toxic = analysis["is_toxic"]
    toxicity_score = analysis["toxicity_score"]

    # 3. İş Aksiyonları (Sansür ve Uyarı)
    censored_text = message.text
    action_taken = "passed"

    if is_toxic:
        censored_text = censor_message(message.text)
        user.warning_count += 1
        action_taken = "warned"
        
        # 3 Uyarıda banla
        if user.warning_count >= 3:
            user.is_banned = True
            action_taken = "banned"
        
        db.commit()

    # 4. Mesajı Logla
    msg_log = models.MessageLog(
        user_id=user.id,
        original_text=message.text,
        censored_text=censored_text,
        is_toxic=is_toxic,
        toxicity_score=toxicity_score
    )
    db.add(msg_log)
    db.commit()

    # 5. İstemciye yanıt dön
    return schemas.MessageResponse(
        status="toxic" if is_toxic else "clean",
        original_text=message.text,
        censored_text=censored_text,
        warning_count=user.warning_count,
        action_taken=action_taken
    )

@app.get("/api/users/{user_id}/status", response_model=schemas.UserStatusResponse)
def get_user_status(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")
    
    return schemas.UserStatusResponse(
        user_id=user.id,
        username=user.username,
        warning_count=user.warning_count,
        is_banned=user.is_banned
    )
