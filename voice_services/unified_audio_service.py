"""
Unified Audio Service
Combines transcription and text-to-speech functionality for voice processing
Integrated directly into Devia backend
"""

import logging
from typing import Optional, Dict, Any
from .audio_transcription_service import AudioTranscriptionService
from .text_to_speech_service import TextToSpeechService
from config.settings import Settings

class UnifiedAudioService:
    """
    Unified service that combines transcription and TTS capabilities
    Provides a single interface for all audio processing needs
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the unified audio service
        
        Args:
            api_key: OpenAI API key (if not provided, will load from settings)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load API key from settings if not provided
        if api_key is None:
            settings = Settings()
            api_key = settings.openai_api_key
            
            if not settings.validate_openai_key():
                self.logger.error("OpenAI API key not found in settings or invalid")
                raise ValueError("OpenAI API key must be configured in settings")
        
        # Initialize sub-services
        try:
            self.transcription_service = AudioTranscriptionService(api_key)
            self.tts_service = TextToSpeechService(api_key)
            self._initialized = True
            self.logger.info("Unified Audio Service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Unified Audio Service: {e}")
            self._initialized = False
            raise
    
    def is_initialized(self) -> bool:
        """Check if the service is properly initialized"""
        return self._initialized
    
    # Transcription methods
    async def transcribe_file(
        self, 
        filepath: str, 
        language: Optional[str] = None, 
        prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file to text
        
        Args:
            filepath: Path to audio file
            language: Optional language code
            prompt: Optional prompt to guide transcription
            
        Returns:
            Transcription result with metadata
        """
        return await self.transcription_service.transcribe_file(filepath, language=language, prompt=prompt)
    
    async def transcribe_bytes(
        self, 
        audio_bytes: bytes, 
        filename: str = "audio.mp3",
        language: Optional[str] = None, 
        prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio from bytes data
        
        Args:
            audio_bytes: Raw audio data
            filename: Filename for format detection
            language: Optional language code
            prompt: Optional prompt to guide transcription
            
        Returns:
            Transcription result with metadata
        """
        return await self.transcription_service.transcribe_bytes(
            audio_bytes, filename, language=language, prompt=prompt
        )
    
    # TTS methods
    async def synthesize_text(
        self, 
        text: str, 
        voice: str = "alloy", 
        output_path: Optional[str] = None,
        model: str = "tts-1"
    ) -> Optional[Dict[str, Any]]:
        """
        Convert text to speech audio file
        
        Args:
            text: Text to convert
            voice: Voice to use
            output_path: Where to save audio file
            model: TTS model to use
            
        Returns:
            Synthesis result with metadata
        """
        return await self.tts_service.synthesize(text, voice, output_path, model)
    
    async def synthesize_to_bytes(
        self, 
        text: str, 
        voice: str = "alloy", 
        model: str = "tts-1"
    ) -> Optional[Dict[str, Any]]:
        """
        Convert text to speech and return as bytes
        
        Args:
            text: Text to convert
            voice: Voice to use
            model: TTS model to use
            
        Returns:
            Synthesis result with audio bytes
        """
        return await self.tts_service.synthesize_to_bytes(text, voice, model)
    
    # Combined workflow methods
    async def voice_to_voice_processing(
        self, 
        audio_bytes: bytes, 
        text_processor_func,
        language: Optional[str] = None,
        transcription_prompt: Optional[str] = None,
        tts_voice: str = "alloy",
        tts_model: str = "tts-1"
    ) -> Dict[str, Any]:
        """
        Complete voice-to-voice processing workflow
        
        Args:
            audio_bytes: Input audio data
            text_processor_func: Function to process transcribed text
            language: Language for transcription
            transcription_prompt: Prompt for transcription
            tts_voice: Voice for TTS output
            tts_model: TTS model to use
            
        Returns:
            Complete processing result including transcription, processed text, and audio output
        """
        try:
            # Step 1: Transcribe audio
            self.logger.info("Starting voice-to-voice processing: transcription")
            transcription_result = await self.transcribe_bytes(
                audio_bytes, 
                language=language, 
                prompt=transcription_prompt
            )
            
            if not transcription_result or not transcription_result.get("success"):
                return {
                    "success": False,
                    "error": "Transcription failed",
                    "transcription_result": transcription_result
                }
            
            transcribed_text = transcription_result["text"]
            
            # Step 2: Process transcribed text
            self.logger.info("Processing transcribed text")
            try:
                processed_result = await text_processor_func(transcribed_text)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Text processing failed: {str(e)}",
                    "transcribed_text": transcribed_text,
                    "transcription_result": transcription_result
                }
            
            # Step 3: Generate human-friendly response for TTS
            if hasattr(processed_result, 'get') and processed_result.get("human_response"):
                tts_text = processed_result["human_response"]
            elif isinstance(processed_result, str):
                tts_text = processed_result
            else:
                tts_text = "I've processed your request successfully. Please check the detailed response."
            
            # Step 4: Convert response to speech
            self.logger.info("Converting response to speech")
            tts_result = await self.synthesize_text(
                tts_text, 
                voice=tts_voice, 
                model=tts_model
            )
            
            if not tts_result or not tts_result.get("success"):
                return {
                    "success": True,  # Still success since processing worked
                    "warning": "TTS generation failed",
                    "transcribed_text": transcribed_text,
                    "processed_result": processed_result,
                    "transcription_result": transcription_result,
                    "tts_result": tts_result
                }
            
            # Return complete result
            return {
                "success": True,
                "transcribed_text": transcribed_text,
                "processed_result": processed_result,
                "tts_text": tts_text,
                "audio_output_path": tts_result.get("output_path"),
                "transcription_result": transcription_result,
                "tts_result": tts_result
            }
            
        except Exception as e:
            self.logger.error(f"Voice-to-voice processing failed: {e}")
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}"
            }
    
    # Service information methods
    def get_supported_audio_formats(self) -> list:
        """Get supported audio formats for transcription"""
        return self.transcription_service.get_supported_formats()
    
    def get_available_voices(self) -> list:
        """Get available TTS voices"""
        return self.tts_service.get_available_voices()
    
    def get_available_tts_models(self) -> list:
        """Get available TTS models"""
        return self.tts_service.get_available_models()
    
    def get_voice_description(self, voice: str) -> str:
        """Get description of a TTS voice"""
        return self.tts_service.get_voice_description(voice)
    
    # Health check methods
    async def test_transcription_connection(self) -> bool:
        """Test transcription service connection"""
        return await self.transcription_service.test_connection()
    
    async def test_tts_connection(self) -> bool:
        """Test TTS service connection"""
        return await self.tts_service.test_connection()
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """Test all service connections"""
        return {
            "transcription": await self.test_transcription_connection(),
            "tts": await self.test_tts_connection(),
            "unified_service": self.is_initialized()
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information"""
        return {
            "initialized": self.is_initialized(),
            "supported_audio_formats": self.get_supported_audio_formats(),
            "available_voices": self.get_available_voices(),
            "available_tts_models": self.get_available_tts_models(),
            "max_audio_file_size_mb": self.transcription_service.get_max_file_size_mb(),
            "voice_descriptions": {
                voice: self.get_voice_description(voice) 
                for voice in self.get_available_voices()
            }
        }