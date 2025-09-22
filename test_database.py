#!/usr/bin/env python3
"""
Database connection test script
Run this to verify database connection works before starting the main application
"""

import asyncio
import logging
import sys
from database import connect_to_mongo, close_mongo_connection, is_connected, get_database
from config.settings import Settings

# Setup logging with UTF-8 encoding for Windows compatibility
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler]
)

logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connection and operations"""
    try:
        logger.info("[TEST] Starting database connection test...")
        
        # Test connection
        await connect_to_mongo()
        
        if is_connected():
            logger.info("[TEST] Database connection test passed!")
            
            # Test basic operations
            db = get_database()
            
            # List collections
            collections = await db.list_collection_names()
            logger.info(f"[TEST] Available collections: {collections}")
            
            # Test a simple operation on each collection
            for collection_name in ['clients', 'invoices', 'quotes', 'expenses']:
                collection = db[collection_name]
                count = await collection.count_documents({})
                logger.info(f"[TEST] {collection_name}: {count} documents")
            
            logger.info("[TEST] All database operations completed successfully!")
            
        else:
            logger.error("[TEST] Database connection test failed!")
            return False
            
    except Exception as e:
        logger.error(f"[TEST] Database connection test error: {str(e)}")
        return False
        
    finally:
        await close_mongo_connection()
        logger.info("[TEST] Database connection closed")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_database_connection())
    if success:
        print("\n[TEST] Database is ready for the application!")
    else:
        print("\n[TEST] Database connection failed. Please check your configuration.")
        exit(1)