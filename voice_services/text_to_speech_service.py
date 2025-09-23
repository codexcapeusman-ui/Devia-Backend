"""
Internal Text-to-Speech Service
Provides text-to-speech functionality using OpenAI TTS API
Integrated directly into Devia backend for voice processing
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from openai import OpenAI, OpenAIError
from config.settings import Settings

class TextToSpeechService:
    """
    Service for converting text to speech using OpenAI TTS API
    Supports multiple voices and provides high-quality audio output
    """
    
    # Available TTS models
    AVAILABLE_MODELS = ["tts-1", "tts-1-hd"]
    DEFAULT_MODEL = "tts-1"  # Fast, good quality
    HD_MODEL = "tts-1-hd"   # Higher quality, slower
    
    # Available voices
    AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    DEFAULT_VOICE = "alloy"
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = DEFAULT_MODEL):
        """
        Initialize the text-to-speech service
        
        Args:
            api_key: OpenAI API key (if not provided, will load from settings)
            default_model: Default TTS model to use
        """
        self.logger = logging.getLogger(__name__)
        
        # Load API key from settings if not provided
        if api_key is None:
            settings = Settings()
            api_key = settings.openai_api_key
            
            if not settings.validate_openai_key():
                raise RuntimeError("OpenAI API key not found in settings or invalid")
        
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = default_model if default_model in self.AVAILABLE_MODELS else self.DEFAULT_MODEL
        
        self.logger.info(f"Text-to-Speech Service initialized with model: {self.default_model}")
    
    async def synthesize(
        self, 
        text: str, 
        voice: str = DEFAULT_VOICE, 
        output_path: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Convert text to speech and save as audio file
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            output_path: Path to save the audio file (if None, generates a name)
            model: TTS model to use (tts-1 or tts-1-hd)
        
        Returns:
            Dict containing synthesis result and metadata, or None if error
        """
        try:
            # Validate inputs
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "Text cannot be empty",
                    "output_path": None
                }
            
            # Set defaults
            model = model or self.default_model
            if voice not in self.AVAILABLE_VOICES:
                self.logger.warning(f"Voice '{voice}' not in available voices: {self.AVAILABLE_VOICES}")
                voice = self.DEFAULT_VOICE
            
            if model not in self.AVAILABLE_MODELS:
                self.logger.warning(f"Model '{model}' not in available models: {self.AVAILABLE_MODELS}")
                model = self.default_model
            
            # Generate output path if not provided
            if not output_path:
                import time
                timestamp = int(time.time())
                output_path = f"temp/tts_output_{timestamp}.mp3"
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Synthesizing text to speech: {len(text)} characters, voice: {voice}, model: {model}")
            
            # Prepare TTS parameters
            tts_params = {
                "model": model,
                "input": text,
                "voice": voice
            }
            
            # Call OpenAI TTS API
            response = self.client.audio.speech.create(**tts_params)
            
            # Save audio to file
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            # Return successful result
            result = {
                "success": True,
                "model": model,
                "voice": voice,
                "output_path": output_path,
                "text_length": len(text),
                "file_size_bytes": os.path.getsize(output_path) if os.path.exists(output_path) else 0
            }
            
            self.logger.info(f"TTS synthesis successful: {output_path}")
            return result
            
        except OpenAIError as e:
            self.logger.error(f"OpenAI API error during TTS: {e}")
            return {
                "success": False,
                "error": f"OpenAI API error: {str(e)}",
                "output_path": None
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during TTS: {e}")
            return {
                "success": False,
                "error": f"TTS error: {str(e)}",
                "output_path": None
            }
    
    async def synthesize_to_bytes(
        self, 
        text: str, 
        voice: str = DEFAULT_VOICE, 
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Convert text to speech and return as bytes
        
        Args:
            text: Text to convert to speech
            voice: Voice to use
            model: TTS model to use
        
        Returns:
            Dict containing audio bytes and metadata
        """
        try:
            # Validate inputs
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "Text cannot be empty",
                    "audio_bytes": None
                }
            
            # Set defaults
            model = model or self.default_model
            if voice not in self.AVAILABLE_VOICES:
                voice = self.DEFAULT_VOICE
            if model not in self.AVAILABLE_MODELS:
                model = self.default_model
            
            self.logger.info(f"Synthesizing text to bytes: {len(text)} characters")
            
            # Prepare TTS parameters
            tts_params = {
                "model": model,
                "input": text,
                "voice": voice
            }
            
            # Call OpenAI TTS API
            response = self.client.audio.speech.create(**tts_params)
            
            # Return audio bytes
            result = {
                "success": True,
                "model": model,
                "voice": voice,
                "audio_bytes": response.content,
                "text_length": len(text),
                "file_size_bytes": len(response.content)
            }
            
            self.logger.info(f"TTS synthesis to bytes successful: {len(response.content)} bytes")
            return result
            
        except OpenAIError as e:
            self.logger.error(f"OpenAI API error during TTS to bytes: {e}")
            return {
                "success": False,
                "error": f"OpenAI API error: {str(e)}",
                "audio_bytes": None
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during TTS to bytes: {e}")
            return {
                "success": False,
                "error": f"TTS error: {str(e)}",
                "audio_bytes": None
            }
    
    async def synthesize_multiple(
        self, 
        texts: List[str], 
        voice: str = DEFAULT_VOICE,
        model: Optional[str] = None,
        output_dir: str = "temp"
    ) -> List[Dict[str, Any]]:
        """
        Convert multiple texts to speech
        
        Args:
            texts: List of texts to convert
            voice: Voice to use for all texts
            model: TTS model to use
            output_dir: Directory to save audio files
        
        Returns:
            List of synthesis results
        """
        results = []
        
        for i, text in enumerate(texts):
            output_path = f"{output_dir}/tts_batch_{i+1}.mp3"
            result = await self.synthesize(text, voice, output_path, model)
            results.append(result)
        
        return results
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voices
        
        Returns:
            List of available voice names
        """
        return self.AVAILABLE_VOICES.copy()
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available TTS models
        
        Returns:
            List of available model names
        """
        return self.AVAILABLE_MODELS.copy()
    
    def get_voice_description(self, voice: str) -> str:
        """
        Get description of a voice
        
        Args:
            voice: Voice name
            
        Returns:
            Description of the voice characteristics
        """
        voice_descriptions = {
            "alloy": "Neutral, balanced voice suitable for most content",
            "echo": "Clear, articulate voice with professional tone",
            "fable": "Warm, storytelling voice with expressive qualities",
            "onyx": "Deep, authoritative voice with strong presence",
            "nova": "Friendly, approachable voice with modern appeal",
            "shimmer": "Bright, energetic voice with youthful qualities"
        }
        
        return voice_descriptions.get(voice, "Voice description not available")
    
    async def test_connection(self) -> bool:
        """
        Test connection to OpenAI TTS API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with a simple phrase
            test_text = "Hello, this is a test."
            result = await self.synthesize_to_bytes(test_text)
            
            if result and result.get("success"):
                self.logger.info("TTS API connection test successful")
                return True
            else:
                self.logger.error("TTS API connection test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"TTS connection test failed: {e}")
            return False
    
    def estimate_audio_duration(self, text: str, words_per_minute: int = 150) -> float:
        """
        Estimate audio duration based on text length
        
        Args:
            text: Text to estimate duration for
            words_per_minute: Average speaking rate
            
        Returns:
            Estimated duration in seconds
        """
        word_count = len(text.split())
        duration_minutes = word_count / words_per_minute
        return duration_minutes * 60
    
    def validate_text_length(self, text: str, max_length: int = 4096) -> Dict[str, Any]:
        """
        Validate text length for TTS processing
        
        Args:
            text: Text to validate
            max_length: Maximum allowed character length
            
        Returns:
            Validation result
        """
        if len(text) > max_length:
            return {
                "valid": False,
                "error": f"Text length ({len(text)}) exceeds maximum ({max_length})",
                "suggestion": "Consider splitting the text into smaller chunks"
            }
        
        return {
            "valid": True,
            "length": len(text),
            "estimated_duration": self.estimate_audio_duration(text)
        }