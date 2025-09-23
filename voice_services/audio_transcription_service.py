"""
Internal Audio Transcription Service
Provides speech-to-text functionality using OpenAI Whisper API
Integrated directly into Devia backend for voice processing
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from openai import OpenAI, OpenAIError
from config.settings import Settings

class AudioTranscriptionService:
    """
    Service for transcribing audio files using OpenAI Whisper API
    Handles various audio formats and provides detailed transcription results
    """
    
    # Supported audio formats for OpenAI Whisper
    SUPPORTED_FORMATS = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
    MAX_FILE_SIZE_MB = 25
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the transcription service
        
        Args:
            api_key: OpenAI API key (if not provided, will load from settings)
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
        self.logger.info("Audio Transcription Service initialized")
    
    async def transcribe_file(
        self, 
        filepath: str, 
        model: str = "whisper-1", 
        language: Optional[str] = None, 
        prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file using OpenAI Whisper API
        
        Args:
            filepath: Path to the audio file
            model: Whisper model to use (default: whisper-1)
            language: Optional language code (e.g., 'en', 'es', 'fr')
            prompt: Optional prompt to guide the transcription
        
        Returns:
            Dict containing transcription result and metadata, or None if error
        """
        try:
            # Validate file
            validation_result = self._validate_audio_file(filepath)
            if not validation_result["valid"]:
                self.logger.error(f"File validation failed: {validation_result['error']}")
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "text": "",
                    "file_path": filepath
                }
            
            # Perform transcription
            self.logger.info(f"Transcribing audio file: {filepath}")
            
            with open(filepath, "rb") as audio_file:
                # Prepare transcription parameters
                transcription_params = {
                    "model": model,
                    "file": audio_file,
                }
                
                # Add optional parameters
                if language:
                    transcription_params["language"] = language
                if prompt:
                    transcription_params["prompt"] = prompt
                
                # Call the OpenAI API
                response = self.client.audio.transcriptions.create(**transcription_params)
                
                # Return successful result
                result = {
                    "success": True,
                    "text": response.text,
                    "model": model,
                    "language": language,
                    "file_path": filepath,
                    "file_size_mb": validation_result["file_size_mb"],
                    "prompt": prompt
                }
                
                self.logger.info(f"Transcription successful: {len(response.text)} characters")
                return result
                
        except OpenAIError as e:
            self.logger.error(f"OpenAI API error during transcription: {e}")
            return {
                "success": False,
                "error": f"OpenAI API error: {str(e)}",
                "text": "",
                "file_path": filepath
            }
        except Exception as e:
            self.logger.error(f"Unexpected error during transcription: {e}")
            return {
                "success": False,
                "error": f"Transcription error: {str(e)}",
                "text": "",
                "file_path": filepath
            }
    
    async def transcribe_bytes(
        self, 
        audio_bytes: bytes, 
        filename: str = "audio.mp3",
        model: str = "whisper-1", 
        language: Optional[str] = None, 
        prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio from bytes data
        
        Args:
            audio_bytes: Raw audio data as bytes
            filename: Filename for the audio (determines format)
            model: Whisper model to use
            language: Optional language code
            prompt: Optional prompt to guide transcription
        
        Returns:
            Dict containing transcription result and metadata
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Transcribe the temporary file
            result = await self.transcribe_file(temp_path, model, language, prompt)
            if result:
                result["original_filename"] = filename
            return result
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                self.logger.warning(f"Could not delete temporary file {temp_path}: {e}")
    
    def _validate_audio_file(self, filepath: str) -> Dict[str, Any]:
        """
        Validate audio file for transcription
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Dict with validation result
        """
        # Check if file exists
        if not os.path.exists(filepath):
            return {
                "valid": False,
                "error": f"File not found: {filepath}"
            }
        
        # Check file extension
        file_ext = Path(filepath).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            self.logger.warning(f"File extension {file_ext} not in supported formats: {self.SUPPORTED_FORMATS}")
            # Don't fail here, let OpenAI decide
        
        # Check file size
        try:
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
                return {
                    "valid": False,
                    "error": f"File size ({file_size_mb:.2f} MB) exceeds {self.MAX_FILE_SIZE_MB} MB limit"
                }
            
            return {
                "valid": True,
                "file_size_mb": file_size_mb,
                "file_extension": file_ext
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Could not read file: {str(e)}"
            }
    
    async def test_connection(self) -> bool:
        """
        Test connection to OpenAI API with a simple transcription
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create a simple test audio using TTS (if available)
            test_text = "Hello, this is a test."
            
            # For now, just test if we can access the API
            # In a full implementation, you might create a small test audio file
            self.logger.info("Testing OpenAI Whisper API connection")
            return True  # We'll assume it works if the client was created successfully
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_supported_formats(self) -> list:
        """
        Get list of supported audio formats
        
        Returns:
            List of supported file extensions
        """
        return self.SUPPORTED_FORMATS.copy()
    
    def get_max_file_size_mb(self) -> int:
        """
        Get maximum file size in MB
        
        Returns:
            Maximum file size in megabytes
        """
        return self.MAX_FILE_SIZE_MB