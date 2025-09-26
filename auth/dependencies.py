"""
FastAPI dependencies for JWT authentication in AI services
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .jwt_auth import get_current_user_id

security = HTTPBearer()

async def get_current_user_id_dependency(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    FastAPI dependency to extract user_id from JWT token
    
    Args:
        credentials: HTTP Bearer credentials containing JWT token
        
    Returns:
        User ID string
        
    Raises:
        HTTPException: If token is invalid or user_id not found
    """
    import logging
    logger = logging.getLogger(__name__)
    
    token = credentials.credentials
    logger.info(f"Received token: {token[:20]}...")  # Log first 20 chars for debugging
    
    # Get user_id from token
    user_id = get_current_user_id(token)
    if user_id is None:
        logger.error("Failed to extract user_id from token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"Successfully extracted user_id: {user_id}")
    return user_id