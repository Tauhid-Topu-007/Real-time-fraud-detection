#!/usr/bin/env python
"""
Run the fraud detection API
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
import yaml
from loguru import logger

def main():
    """Main entry point"""
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("🚀 Starting Fraud Detection API...")
    logger.info(f"Host: {config['api']['host']}")
    logger.info(f"Port: {config['api']['port']}")
    
    uvicorn.run(
        "src.api.main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['reload']
    )

if __name__ == "__main__":
    main()