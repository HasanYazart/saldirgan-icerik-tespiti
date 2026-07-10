from pydantic import BaseModel
from datetime import datetime

class MessageCreate(BaseModel):
    user_id: int
    username: str  # Kullanıcı daha önce yoksa otomatik oluşturmak için
    text: str

class MessageResponse(BaseModel):
    status: str
    original_text: str
    censored_text: str
    warning_count: int
    action_taken: str

class UserStatusResponse(BaseModel):
    user_id: int
    username: str
    warning_count: int
    is_banned: bool
