"""
State Backup Module
Handles automatic backups of bot state with rotation and recovery.
"""
import os
import json
import shutil
import logging
from typing import Optional, List
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class StateBackup:
    """Manages automatic backups of bot state"""
    
    BACKUP_DIR = os.path.join(config.DATA_DIR, "backups")
    MAX_BACKUPS = 10  # Keep last 10 backups
    
    def __init__(self):
        """Initialize backup system"""
        os.makedirs(self.BACKUP_DIR, exist_ok=True)
    
    def create_backup(self, state_file: str, backup_type: str = "auto") -> Optional[str]:
        """
        Create a backup of the state file
        
        Args:
            state_file: Path to state file to backup
            backup_type: Type of backup ("auto", "manual", "pre_trade", "pre_shutdown")
        
        Returns:
            Path to backup file, or None if backup failed
        """
        if not os.path.exists(state_file):
            logger.warning(f"Cannot backup {state_file}: file does not exist")
            return None
        
        try:
            # Create timestamped backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(state_file)
            backup_name = f"{filename}.{backup_type}.{timestamp}.bak"
            backup_path = os.path.join(self.BACKUP_DIR, backup_name)
            
            # Copy file
            shutil.copy2(state_file, backup_path)
            logger.info(f"Created backup: {backup_name} ({backup_type})")
            
            # Rotate old backups
            self._rotate_backups(filename)
            
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def _rotate_backups(self, base_filename: str):
        """Remove old backups, keeping only the most recent N"""
        try:
            # Find all backups for this file
            backups = []
            for file in os.listdir(self.BACKUP_DIR):
                if file.startswith(base_filename + "."):
                    file_path = os.path.join(self.BACKUP_DIR, file)
                    backups.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            if len(backups) > self.MAX_BACKUPS:
                for backup_path, _ in backups[self.MAX_BACKUPS:]:
                    try:
                        os.remove(backup_path)
                        logger.debug(f"Removed old backup: {os.path.basename(backup_path)}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old backup {backup_path}: {e}")
        except Exception as e:
            logger.warning(f"Error rotating backups: {e}")
    
    def list_backups(self, state_file: str) -> List[dict]:
        """List all backups for a state file"""
        backups = []
        filename = os.path.basename(state_file)
        
        try:
            for file in os.listdir(self.BACKUP_DIR):
                if file.startswith(filename + "."):
                    file_path = os.path.join(self.BACKUP_DIR, file)
                    stat = os.stat(file_path)
                    backups.append({
                        "filename": file,
                        "path": file_path,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x["modified"], reverse=True)
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
        
        return backups
    
    def restore_backup(self, backup_path: str, target_file: str) -> bool:
        """
        Restore a backup file
        
        Args:
            backup_path: Path to backup file
            target_file: Path where backup should be restored
        
        Returns:
            True if restore successful, False otherwise
        """
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Create backup of current file before restore
            if os.path.exists(target_file):
                self.create_backup(target_file, "pre_restore")
            
            # Restore backup
            shutil.copy2(backup_path, target_file)
            logger.info(f"Restored backup: {os.path.basename(backup_path)} -> {target_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def get_latest_backup(self, state_file: str) -> Optional[str]:
        """Get path to latest backup for a state file"""
        backups = self.list_backups(state_file)
        if backups:
            return backups[0]["path"]
        return None


# Singleton instance
_backup_manager: Optional[StateBackup] = None


def get_backup_manager() -> StateBackup:
    """Get singleton backup manager instance"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = StateBackup()
    return _backup_manager
