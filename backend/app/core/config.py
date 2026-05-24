from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/creditai"
    
    # JWT
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Currency
    CURRENCY: str = "INR"
    CURRENCY_SYMBOL: str = "₹"
    
    # API
    API_VERSION: str = "v1"
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()