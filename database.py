from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from config.settings import Settings
import asyncio
import logging

# Setup logging
logger = logging.getLogger(__name__)

settings = Settings()

class Database:
    client: AsyncIOMotorClient = None
    database = None
    _connection_tested = False

db = Database()

# MongoDB connection
async def connect_to_mongo():
    """Create database connection with proper logging and error handling"""
    try:
        logger.info("[DB] Initializing MongoDB connection...")
        logger.info(f"[DB] Connecting to: {settings.database_url}")
        
        db.client = AsyncIOMotorClient(settings.database_url)
        db.database = db.client[settings.database_name]
        
        # Test the connection
        await test_connection()
        
        logger.info(f"[DB] MongoDB connection established successfully!")
        logger.info(f"[DB] Connected to database: {settings.database_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
        logger.error(f"🔧 Database URL: {settings.database_url}")
        logger.error(f"🗄️ Database Name: {settings.database_name}")
        raise e

async def test_connection():
    """Test database connection and log database info"""
    try:
        # Test connection with a simple ping
        await db.client.admin.command('ping')
        db._connection_tested = True
        
        # Get server info
        server_info = await db.client.server_info()
        logger.info(f"[DB] MongoDB Server Version: {server_info.get('version', 'Unknown')}")
        
        # List existing collections
        collections = await db.database.list_collection_names()
        if collections:
            logger.info(f"[DB] Existing collections: {', '.join(collections)}")
        else:
            logger.info("[DB] No existing collections found - database is ready for new data")
            
        return True
        
    except Exception as e:
        logger.error(f"[DB] Database connection test failed: {str(e)}")
        db._connection_tested = False
        raise e

def is_connected():
    """Check if database is connected"""
    return db.client is not None and db._connection_tested

async def close_mongo_connection():
    """Close database connection with proper logging"""
    try:
        if db.client:
            logger.info("[DB] Closing MongoDB connection...")
            db.client.close()
            db._connection_tested = False
            logger.info("[DB] MongoDB connection closed successfully")
        else:
            logger.warning("[DB] No active MongoDB connection to close")
    except Exception as e:
        logger.error(f"[DB] Error closing MongoDB connection: {str(e)}")

def get_database():
    """Get database instance with connection check"""
    if not is_connected():
        logger.warning("[DB] Database not connected. Please ensure connect_to_mongo() was called.")
    return db.database

# Collections
def get_users_collection():
    return db.database.users

def get_subscriptions_collection():
    return db.database.subscriptions

def get_settings_collection():
    return db.database.user_settings

def get_clients_collection():
    return db.database.clients

def get_expenses_collection():
    return db.database.expenses

def get_invoices_collection():
    return db.database.invoices

def get_quotes_collection():
    return db.database.quotes

def get_jobs_collection():
    return db.database.jobs

def get_meetings_collection():
    return db.database.meetings

def get_manual_tasks_collection():
    return db.database.manual_tasks