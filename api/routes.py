"""
FastAPI routes for unified AI agent interactions
Provides single endpoint for all AI agent operations with intelligent routing
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from services.semantic_kernel_service import SemanticKernelService
from services.unified_agent_service import UnifiedAgentService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
agent_router = APIRouter(prefix="/agent", tags=["AI Agents"])

# Global unified agent service
unified_agent_service = None

class UnifiedAgentRequest(BaseModel):
    """Unified request model for all AI agent interactions"""
    prompt: str
    user_id: str
    language: str = "en"
    context: Optional[Dict[str, Any]] = None

class ConversationStatusRequest(BaseModel):
    """Request model for conversation status"""
    user_id: str

class ConversationResetRequest(BaseModel):
    """Request model for conversation reset"""
    user_id: str

def get_sk_service(request: Request) -> SemanticKernelService:
    """Dependency to get Semantic Kernel service from app state"""
    sk_service = getattr(request.app.state, 'sk_service', None)
    if not sk_service:
        raise HTTPException(status_code=500, detail="Semantic Kernel service not initialized")
    return sk_service

def get_unified_agent_service(sk_service: SemanticKernelService = Depends(get_sk_service)) -> UnifiedAgentService:
    """Dependency to get Unified Agent service"""
    global unified_agent_service
    
    if not unified_agent_service:
        unified_agent_service = UnifiedAgentService(sk_service)
    
    return unified_agent_service

@agent_router.post("/process", response_model=Dict[str, Any])
async def process_agent_request(
    request: UnifiedAgentRequest,
    unified_service: UnifiedAgentService = Depends(get_unified_agent_service)
):
    """
    Unified AI Agent Endpoint
    
    This single endpoint handles all AI agent operations:
    - Invoice generation
    - Quote creation
    - Customer data extraction
    - Job scheduling
    - Expense tracking
    
    Workflow:
    1. Detect intent from user prompt
    2. Extract relevant data
    3. Check for missing required fields
    4. Ask for missing data or return final result
    
    Example requests:
    ```json
    {
        "prompt": "Create invoice for John Doe at ABC Corp for website development 40 hours at €50/hour",
        "user_id": "user123",
        "language": "en"
    }
    
    {
        "prompt": "Add new customer: Jane Smith, email jane@example.com, phone 555-0123",
        "user_id": "user123", 
        "language": "en"
    }
    
    {
        "prompt": "Schedule website maintenance for tomorrow at 2 PM, duration 3 hours",
        "user_id": "user123",
        "language": "en"
    }
    ```
    """
    try:
        logger.info(f"Processing unified agent request for user {request.user_id}: {request.prompt[:100]}...")
        
        # Process the request through unified agent
        result = await unified_service.process_agent_request(
            prompt=request.prompt,
            user_id=request.user_id,
            language=request.language
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Unified agent request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")

@agent_router.post("/conversation/status")
async def get_conversation_status(
    request: ConversationStatusRequest,
    unified_service: UnifiedAgentService = Depends(get_unified_agent_service)
):
    """
    Get current conversation status for a user
    
    Returns information about:
    - Current conversation state
    - Detected intent and confidence
    - Whether data collection is in progress
    - Last update timestamp
    """
    try:
        status = unified_service.get_conversation_status(request.user_id)
        return {
            "success": True,
            "data": status
        }
        
    except Exception as e:
        logger.error(f"Failed to get conversation status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@agent_router.post("/conversation/reset")
async def reset_conversation(
    request: ConversationResetRequest,
    unified_service: UnifiedAgentService = Depends(get_unified_agent_service)
):
    """
    Reset conversation state for a user
    
    Use this to:
    - Start a fresh conversation
    - Clear incomplete data collection
    - Reset intent detection
    """
    try:
        unified_service.reset_conversation(request.user_id)
        return {
            "success": True,
            "message": "Conversation reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to reset conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")

@agent_router.get("/health")
async def agent_health_check(sk_service: SemanticKernelService = Depends(get_sk_service)):
    """
    Check the health status of AI agents
    """
    try:
        health_status = {
            "status": "healthy",
            "semantic_kernel": sk_service.is_initialized(),
            "openai_connection": await sk_service.test_openai_connection(),
            "unified_agent": unified_agent_service is not None,
            "supported_intents": [
                "invoice_generation",
                "quote_creation",
                "customer_extraction", 
                "job_scheduling",
                "expense_tracking"
            ],
            "supported_languages": ["en", "fr"]
        }
        
        if not health_status["semantic_kernel"] or not health_status["openai_connection"]:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@agent_router.post("/test")
async def test_unified_agent(
    request: UnifiedAgentRequest,
    unified_service: UnifiedAgentService = Depends(get_unified_agent_service)
):
    """
    Test endpoint for unified AI agent
    Returns debug information about the processing workflow
    """
    try:
        logger.info(f"Processing test request: {request.prompt[:100]}...")
        
        # Get conversation state before processing
        pre_status = unified_service.get_conversation_status(request.user_id)
        
        # Process the request
        result = await unified_service.process_agent_request(
            prompt=request.prompt,
            user_id=request.user_id,
            language=request.language
        )
        
        # Get conversation state after processing
        post_status = unified_service.get_conversation_status(request.user_id)
        
        # Return debug information
        return {
            "success": True,
            "message": "Test completed successfully",
            "debug_info": {
                "received_prompt": request.prompt,
                "user_id": request.user_id,
                "language": request.language,
                "pre_processing_state": pre_status,
                "post_processing_state": post_status,
                "sk_initialized": unified_service.sk_service.is_initialized(),
                "openai_available": await unified_service.sk_service.test_openai_connection()
            },
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Test request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")