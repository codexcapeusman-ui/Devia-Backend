"""
FastAPI routes for unified AI agent interactions
Provides single endpoint for all AI agent operations with intelligent routing
"""

from fastapi import APIRouter, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import json
import os
import asyncio
import tempfile
from pathlib import Path
import sys

# Add parent directory to path to import transcription and TTS modules
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

from services.semantic_kernel_service import SemanticKernelService
from services.unified_agent_service import UnifiedAgentService
from voice_services.unified_agent_service import UnifiedAgentService as VoiceUnifiedAgentService
from voice_services.semantic_kernel_service import SemanticKernelService as VoiceSemanticKernelService

# Configure logging
logger = logging.getLogger(__name__)

# Import transcription and TTS services from voice_services
try:
    from voice_services.unified_audio_service import UnifiedAudioService
    from voice_services.audio_transcription_service import AudioTranscriptionService
    from voice_services.text_to_speech_service import TextToSpeechService
    audio_services_available = True
    logger.info("Internal audio services imported successfully")
except ImportError as e:
    logger.warning(f"Could not import internal audio services: {e}")
    audio_services_available = False
    UnifiedAudioService = None
    AudioTranscriptionService = None
    TextToSpeechService = None

# Configure logging
logger = logging.getLogger(__name__)

# Create router
agent_router = APIRouter(prefix="/agent", tags=["AI Agents"])

# Global unified agent service
unified_agent_service = None
voice_unified_agent_service = None

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

def get_voice_sk_service(request: Request) -> VoiceSemanticKernelService:
    """Dependency to get Voice Semantic Kernel service from app state"""
    voice_sk_service = getattr(request.app.state, 'voice_sk_service', None)
    if not voice_sk_service:
        raise HTTPException(status_code=500, detail="Voice Semantic Kernel service not initialized")
    return voice_sk_service

def get_unified_agent_service(sk_service: SemanticKernelService = Depends(get_sk_service)) -> UnifiedAgentService:
    """Dependency to get Unified Agent service"""
    global unified_agent_service
    
    if not unified_agent_service:
        unified_agent_service = UnifiedAgentService(sk_service)
    
    return unified_agent_service

def get_voice_unified_agent_service(voice_sk_service: VoiceSemanticKernelService = Depends(get_voice_sk_service)) -> VoiceUnifiedAgentService:
    """Dependency to get Voice Unified Agent service"""
    global voice_unified_agent_service
    
    if not voice_unified_agent_service:
        voice_unified_agent_service = VoiceUnifiedAgentService(voice_sk_service)
    
    return voice_unified_agent_service

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

# WebSocket endpoint for voice processing
@agent_router.websocket("/voice")
async def voice_agent_websocket(
    websocket: WebSocket,
    user_id: str,
    language: str = "en"
):
    """
    WebSocket endpoint for voice-based AI agent interactions
    
    Workflow:
    1. Accept WebSocket connection
    2. Receive audio file from frontend
    3. Transcribe audio using gpt_transcribe.py
    4. Process the transcribed text through voice unified agent
    5. Generate both structured response and human-friendly response
    6. Convert human response to audio using gpt_tts.py
    7. Send both structured data and audio file back to frontend
    
    Parameters:
        user_id: Unique identifier for the user
        language: Language preference (en/fr)
    
    Message Format:
    - Client sends: {"type": "audio", "data": base64_encoded_audio_data, "filename": "audio.mp3"}
    - Server sends: {"type": "response", "structured_data": {...}, "audio_url": "/path/to/audio.mp3", "human_text": "..."}
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for user {user_id}")
    
    try:
        # Check if audio services are available
        if not audio_services_available:
            await websocket.send_json({
                "type": "error",
                "message": "Audio processing services not available. Please check audio service setup."
            })
            await websocket.close()
            return
        
        # Initialize audio and voice services
        voice_unified_service = None
        audio_service = None
        
        try:
            # Create audio service
            audio_service = UnifiedAudioService()
            
            # Create voice services (simplified - in production these should be properly managed)
            from config.settings import Settings
            settings = Settings()
            voice_sk_service = VoiceSemanticKernelService(settings)
            await voice_sk_service.initialize()
            voice_unified_service = VoiceUnifiedAgentService(voice_sk_service)
            logger.info("Voice and audio services initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize voice services: {e}")
            # Will use fallback processing with just audio service
        
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_json()
                
                if message.get("type") == "audio":
                    await process_voice_message(
                        websocket, message, user_id, language, voice_unified_service, audio_service
                    )
                elif message.get("type") == "reset":
                    # Reset conversation state
                    if voice_unified_service:
                        voice_unified_service.reset_conversation(user_id)
                    await websocket.send_json({
                        "type": "reset_confirmation",
                        "message": "Conversation reset successfully"
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message.get('type')}"
                    })
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket connection closed for user {user_id}")
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error processing voice message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing request: {str(e)}"
                })
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        await websocket.close()

async def process_voice_message(
    websocket: WebSocket,
    message: Dict[str, Any],
    user_id: str,
    language: str,
    voice_unified_service: Optional[VoiceUnifiedAgentService],
    audio_service: Optional["UnifiedAudioService"] = None
):
    """
    Process a voice message through the complete workflow
    
    Args:
        websocket: WebSocket connection
        message: Message containing audio data
        user_id: User identifier
        language: Language preference
        voice_unified_service: Voice unified agent service
        audio_service: Audio processing service
    """
    try:
        # Step 1: Extract audio data
        audio_data = message.get("data")
        filename = message.get("filename", "audio.mp3")
        
        if not audio_data:
            await websocket.send_json({
                "type": "error",
                "message": "No audio data provided"
            })
            return
        
        # Step 2: Save audio to temporary file (for compatibility)
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        # If we have voice_unified_service with audio capabilities, use it directly
        if voice_unified_service and hasattr(voice_unified_service, 'process_audio_request'):
            logger.info(f"Using voice unified service for complete audio processing")
            
            result = await voice_unified_service.process_audio_request(
                audio_bytes=audio_bytes,
                user_id=user_id,
                language=language,
                audio_filename=filename
            )
            
            if result and result.get("success"):
                await websocket.send_json({
                    "type": "transcription",
                    "text": result.get("transcribed_text", "")
                })
                
                await websocket.send_json({
                    "type": "response",
                    "structured_data": result.get("structured_response", {}),
                    "human_text": result.get("human_response", ""),
                    "audio_url": f"/agent/audio/{Path(result.get('audio_url', '')).name}" if result.get('audio_url') else None,
                    "transcribed_text": result.get("transcribed_text", "")
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Voice processing failed: {result.get('error', 'Unknown error')}"
                })
            return
        
        # Fallback: Use audio_service for individual steps
        if not audio_service:
            await websocket.send_json({
                "type": "error", 
                "message": "Audio processing service not available"
            })
            return
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Step 3: Transcribe audio using audio service
            logger.info(f"Transcribing audio for user {user_id}")
            transcription_result = await audio_service.transcribe_file(temp_audio_path, language=language)
            
            if not transcription_result or not transcription_result.get("success"):
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to transcribe audio"
                })
                return
            
            transcribed_text = transcription_result["text"]
            logger.info(f"Transcription successful: {transcribed_text[:100]}...")
            
            # Send transcription confirmation
            await websocket.send_json({
                "type": "transcription",
                "text": transcribed_text
            })
            
            # Step 4: Process through voice unified agent (placeholder for now)
            # In a real implementation, this would use the voice_unified_service
            structured_response = await process_with_voice_agent(
                transcribed_text, user_id, language, voice_unified_service
            )
            
            # Step 5: Generate human-friendly response
            human_response = ""
            if voice_unified_service:
                # Use the voice service's enhanced human response generation
                human_response = voice_unified_service.generate_human_friendly_response(structured_response)
            else:
                # Fallback to the simple human response generation
                human_response = generate_human_response(structured_response)
            
            # Step 6: Convert human response to audio using audio service
            audio_filename = f"response_{user_id}_{int(asyncio.get_event_loop().time())}.mp3"
            audio_path = f"temp/{audio_filename}"
            
            # Ensure temp directory exists
            os.makedirs("temp", exist_ok=True)
            
            tts_result = await audio_service.synthesize_text(
                text=human_response,
                voice="alloy",
                output_path=audio_path
            )
            
            if not tts_result or not tts_result.get("success"):
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to generate audio response"
                })
                return
            
            # Step 7: Send complete response
            await websocket.send_json({
                "type": "response",
                "structured_data": structured_response,
                "human_text": human_response,
                "audio_url": f"/agent/audio/{audio_filename}",
                "transcribed_text": transcribed_text
            })
            
        finally:
            # Clean up temporary audio file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
    except Exception as e:
        logger.error(f"Error in process_voice_message: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Error processing voice message: {str(e)}"
        })

async def process_with_voice_agent(
    prompt: str,
    user_id: str,
    language: str,
    voice_unified_service: Optional[VoiceUnifiedAgentService]
) -> Dict[str, Any]:
    """
    Process prompt with voice unified agent service
    
    Args:
        prompt: Transcribed text prompt
        user_id: User identifier
        language: Language preference
        voice_unified_service: Voice unified agent service
        
    Returns:
        Structured response from the agent
    """
    if voice_unified_service:
        try:
            result = await voice_unified_service.process_agent_request(
                prompt=prompt,
                user_id=user_id,
                language=language
            )
            return result
        except Exception as e:
            logger.error(f"Voice agent processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process with voice agent"
            }
    else:
        # Fallback response when voice service is not available
        return {
            "success": True,
            "message": "Voice service not initialized. Using fallback response.",
            "data": {
                "intent": "unknown",
                "prompt": prompt,
                "fallback": True
            }
        }

def generate_human_response(structured_response: Dict[str, Any]) -> str:
    """
    Generate human-friendly response from structured data
    
    Args:
        structured_response: Structured response from the agent
        
    Returns:
        Human-friendly text for TTS conversion
    """
    if not structured_response.get("success"):
        return "I'm sorry, but I encountered an error processing your request. Please try again."
    
    data = structured_response.get("data", {})
    intent = data.get("intent", "unknown")
    
    # Generate response based on intent
    if intent == "invoice":
        customer_name = data.get("customer_name", "the customer")
        total_amount = data.get("total_amount", "the specified amount")
        return f"I've created an invoice for {customer_name} with a total amount of {total_amount}. The invoice has been saved and is ready for review."
    
    elif intent == "quote":
        customer_name = data.get("customer_name", "the customer")
        estimated_total = data.get("estimated_total", "the estimated amount")
        return f"I've prepared a quote for {customer_name} with an estimated total of {estimated_total}. The quote is ready for your review and can be sent to the customer."
    
    elif intent == "customer":
        name = data.get("name", "the new contact")
        return f"I've added {name} to your customer database. Their information has been saved and they're ready for future transactions."
    
    elif intent == "job":
        title = data.get("title", "the job")
        scheduled_date = data.get("scheduled_date", "the scheduled time")
        return f"I've scheduled {title} for {scheduled_date}. The job has been added to your calendar and all relevant details have been saved."
    
    elif intent == "expense":
        description = data.get("description", "the expense")
        amount = data.get("amount", "the amount")
        return f"I've recorded the expense for {description} with an amount of {amount}. This has been added to your expense tracking system."
    
    else:
        return "I've processed your request successfully. Please check the detailed response for more information."

# Endpoint to serve generated audio files
@agent_router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Serve generated audio files
    
    Args:
        filename: Name of the audio file to serve
        
    Returns:
        Audio file response
    """
    audio_path = f"temp/{filename}"
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=filename
    )