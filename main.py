"""
Devia Backend - AI Agent System
Main application entry point with FastAPI server
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

from config.settings import Settings
from api.routes import agent_router
from services.semantic_kernel_service import SemanticKernelService

# Load environment variables
load_dotenv()

# Global service instance
sk_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    global sk_service
    
    # Startup
    print("🚀 Starting Devia AI Agent System...")
    
    # Initialize Semantic Kernel service
    settings = Settings()
    sk_service = SemanticKernelService(settings)
    await sk_service.initialize()
    
    # Store in app state for access in routes
    app.state.sk_service = sk_service
    
    print("✅ AI Agent System initialized successfully!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Devia AI Agent System...")
    if sk_service:
        await sk_service.cleanup()

# Create FastAPI application
app = FastAPI(
    title="Devia AI Agent System",
    description="Semantic Kernel-based AI agents for business automation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(agent_router, prefix="/api")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "errors": [exc.detail]
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "errors": [str(exc)]
        }
    )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Devia AI Agent System is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global sk_service
    
    health_status = {
        "status": "healthy",
        "services": {
            "semantic_kernel": sk_service is not None and sk_service.is_initialized(),
            "openai": sk_service is not None and await sk_service.test_openai_connection()
        }
    }
    
    if not all(health_status["services"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    # Run the application
    settings = Settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info"
    )