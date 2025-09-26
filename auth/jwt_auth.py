"""
JWT authentication utilities for AI services
Based on the core Backend JWT implementation but simplified for AI service needs
"""

from typing import Optional
from jose import JWTError, jwt
from config.settings import Settings

settings = Settings()

def verify_token(token: str, token_type: str = "access") -> Optional[dict]:
    """
    Verify JWT token and return payload
    
    Args:
        token: JWT token string
        token_type: Type of token (default: "access")
        
    Returns:
        Token payload dict if valid, None if invalid
    """
    try:
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Verifying token with SECRET_KEY length: {len(settings.secret_key)}")
        logger.info(f"Algorithm: {settings.algorithm}")
        
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        logger.info(f"Token decoded successfully. Payload: {payload}")
        
        if token_type == "access" and payload.get("type") != "access":
            logger.warning(f"Token type mismatch. Expected: {token_type}, Got: {payload.get('type')}")
            return None
        return payload
    except JWTError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"JWT verification failed: {e}")
        return None

def get_current_user_id(token: str) -> Optional[str]:
    """
    Extract user_id from JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        User ID if token is valid, None if invalid
    """
    payload = verify_token(token, "access")
    if payload is None:
        return None
    
    return payload.get("sub")  # "sub" is the standard JWT claim for user ID