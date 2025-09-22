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
from datetime import datetime

# Setup logging configuration with Windows compatibility
import sys

# Create console handler with UTF-8 encoding for Windows
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create file handler with UTF-8 encoding
file_handler = logging.FileHandler('devia_backend.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)

logger = logging.getLogger(__name__)

from config.settings import Settings
from api.routes import agent_router
from services.semantic_kernel_service import SemanticKernelService
from database import connect_to_mongo, close_mongo_connection, is_connected, get_database

# Load environment variables
load_dotenv()

# Global service instances
sk_service = None
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    global sk_service
    
    # Startup
    logger.info("[STARTUP] Starting Devia AI Agent System...")
    
    try:
        # Initialize database connection first
        logger.info("[STARTUP] Initializing Database Services...")
        await connect_to_mongo()
        
        # Initialize Semantic Kernel service
        logger.info("[STARTUP] Initializing AI Services...")
        sk_service = SemanticKernelService(settings)
        await sk_service.initialize()
        
        # Store in app state for access in routes
        app.state.sk_service = sk_service
        
        logger.info("[STARTUP] All services initialized successfully!")
        logger.info("[STARTUP] Devia AI Agent System is ready to serve requests!")
        
    except Exception as e:
        logger.error(f"[STARTUP] Failed to initialize services: {str(e)}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Devia AI Agent System...")
    
    try:
        if sk_service:
            logger.info("🤖 Cleaning up AI services...")
            await sk_service.cleanup()
        
        # Close database connection
        logger.info("📊 Closing database connection...")
        await close_mongo_connection()
        
        logger.info("✅ Shutdown completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")

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
    """Health check endpoint with startup info"""
    return {
        "message": "Devia AI Agent System is running",
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "database": is_connected(),
            "ai_agent": sk_service is not None and sk_service.is_initialized()
        },
        "startup_time": datetime.now().isoformat()
    }

@app.get("/database/status")
async def database_status():
    """Get detailed database connection status"""
    try:
        if not is_connected():
            return {
                "status": "disconnected",
                "message": "Database is not connected",
                "connected": False
            }
        
        # Test database with a simple operation
        db = get_database()
        collections = await db.list_collection_names()
        
        return {
            "status": "connected",
            "message": "Database is connected and responsive",
            "connected": True,
            "database_name": settings.database_name,
            "collections": collections,
            "collection_count": len(collections)
        }
        
    except Exception as e:
        logger.error(f"Database status check failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "connected": False
        }

@app.get("/health")
async def health_check():
    """Detailed health check with database status"""
    global sk_service
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": is_connected(),
            "semantic_kernel": sk_service is not None and sk_service.is_initialized(),
            "openai": sk_service is not None and await sk_service.test_openai_connection()
        }
    }
    
    # Check if all services are healthy
    if not all(health_status["services"].values()):
        health_status["status"] = "degraded"
        
        # Log which services are down
        for service, status in health_status["services"].items():
            if not status:
                logger.warning(f"⚠️ Service '{service}' is not healthy")
    
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