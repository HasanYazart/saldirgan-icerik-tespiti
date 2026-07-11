"""DATA_ENCRYPTION_KEY için güvenli bir Fernet anahtarı üretir."""

from cryptography.fernet import Fernet

print(Fernet.generate_key().decode("ascii"))
