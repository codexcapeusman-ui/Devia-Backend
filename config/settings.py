"""
Application settings and configuration
"""

from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"
    
    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    
    # Semantic Kernel Configuration
    sk_log_level: str = "INFO"
    
    # Business Configuration
    default_vat_rate: float = 20.0
    default_currency: str = "EUR"
    company_name: str = "Devia"
    company_address: str = "123 Business Street, Paris, France"
    company_email: str = "contact@devia.com"
    company_phone: str = "+33 1 23 45 67 89"
    
    # API Configuration
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as a list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    def validate_openai_key(self) -> bool:
        """Validate that OpenAI API key is set"""
        return bool(self.openai_api_key and self.openai_api_key != "your_openai_api_key_here")