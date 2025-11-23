#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean application data and reset to fresh deployment state.
This script removes all usage data to simulate a fresh Docker deployment.
"""
import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_vector_database():
    """Remove all vector database files to reset to fresh state."""
    
    logger.info("🧹 Cleaning vector database...")
    
    vector_db_paths = [
        "vector_db",  # Main vector database
        "open-webui/vector_db"  # Open WebUI vector database if exists
    ]
    
    for db_path in vector_db_paths:
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path)
                logger.info(f"✅ Removed: {db_path}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {db_path}: {e}")
        else:
            logger.info(f"⏭️  Not found: {db_path}")

def clean_open_webui_data():
    """Clean Open WebUI data including uploaded files and cache."""
    
    logger.info("🧹 Cleaning Open WebUI data...")
    
    open_webui_paths = [
        "open-webui/uploads",      # Uploaded files
        "open-webui/cache",        # Cache data
        "open-webui/webui.db",     # Database file
    ]
    
    for path in open_webui_paths:
        if os.path.exists(path):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    logger.info(f"✅ Removed file: {path}")
                else:
                    shutil.rmtree(path)
                    logger.info(f"✅ Removed directory: {path}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {path}: {e}")
        else:
            logger.info(f"⏭️  Not found: {path}")

def clean_uploaded_json_files():
    """Remove any JSON files that were uploaded/created after original data."""
    
    logger.info("🧹 Cleaning uploaded JSON files...")
    
    data_path = Path("data")
    if not data_path.exists():
        logger.info("⏭️  Data folder not found")
        return
    
    # Original data files (keep these)
    original_files = {
        "2011.json", "2011cgi.json", "2012.json", "2013.json", "2014.json",
        "2015.json", "2016.json", "2017.json", "2018.json", "2019part2.json",
        "2019partie1.json", "2020part1.json", "2020part2,3.json", "2021.json",
        "2022.json", "2023.json", "2024.json", "2025.json"
    }
    
    removed_count = 0
    for json_file in data_path.glob("*.json"):
        if json_file.name not in original_files:
            try:
                json_file.unlink()
                logger.info(f"✅ Removed uploaded file: {json_file.name}")
                removed_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to remove {json_file.name}: {e}")
    
    if removed_count == 0:
        logger.info("⏭️  No uploaded JSON files found to remove")

def clean_temp_files():
    """Clean any temporary files and test files."""
    
    logger.info("🧹 Cleaning temporary files...")
    
    temp_patterns = [
        "*.tmp",
        "test_*.pdf",
        "sample_*.pdf",
        "temp_*.json",
        "*.txt"  # Test text files
    ]
    
    removed_count = 0
    for pattern in temp_patterns:
        for file in Path(".").glob(pattern):
            if file.is_file():
                try:
                    file.unlink()
                    logger.info(f"✅ Removed temp file: {file.name}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to remove {file.name}: {e}")
    
    if removed_count == 0:
        logger.info("⏭️  No temporary files found to remove")

def reset_to_fresh_state():
    """Reset application to fresh deployment state."""
    
    logger.info("🚀 Resetting application to fresh deployment state...")
    logger.info("="*60)
    
    # Clean all data
    clean_vector_database()
    clean_open_webui_data()
    clean_uploaded_json_files()
    clean_temp_files()
    
    logger.info("="*60)
    logger.info("✅ Application reset complete!")
    logger.info("📋 Summary:")
    logger.info("   • Vector database: Reset to empty")
    logger.info("   • Open WebUI data: Cleaned")
    logger.info("   • Uploaded files: Removed")
    logger.info("   • Original data files: Preserved")
    logger.info("")
    logger.info("🔄 Next steps:")
    logger.info("   1. Restart Docker containers")
    logger.info("   2. Test fresh indexing from original data")
    logger.info("   3. Test document upload functionality")

def create_backup():
    """Create backup of current state before cleaning."""
    
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backup_{timestamp}")
    
    logger.info(f"💾 Creating backup: {backup_dir}")
    
    backup_items = ["vector_db", "open-webui", "data"]
    
    for item in backup_items:
        if os.path.exists(item):
            try:
                if os.path.isfile(item):
                    backup_dir.mkdir(exist_ok=True)
                    shutil.copy2(item, backup_dir / item)
                else:
                    shutil.copytree(item, backup_dir / item, ignore_errors=True)
                logger.info(f"✅ Backed up: {item}")
            except Exception as e:
                logger.error(f"❌ Failed to backup {item}: {e}")
    
    logger.info(f"💾 Backup created at: {backup_dir}")
    return backup_dir

if __name__ == "__main__":
    print("🏗️  Fresh Application State Reset Tool")
    print("="*60)
    
    # Ask for confirmation
    response = input("⚠️  This will reset the application to fresh state. Continue? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        # Create backup first
        try:
            backup_dir = create_backup()
            print(f"💾 Backup created: {backup_dir}")
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            response = input("Continue without backup? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Operation cancelled")
                exit(1)
        
        # Reset to fresh state
        reset_to_fresh_state()
        
        print("\n🎯 Ready for fresh deployment testing!")
        print("Run: docker-compose up -d --build")
        
    else:
        print("❌ Operation cancelled")