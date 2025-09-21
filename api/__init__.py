"""
API package for Devia Backend
Exposes REST API routes for AI agent interactions
"""

from .routes import agent_router

__all__ = ["agent_router"]