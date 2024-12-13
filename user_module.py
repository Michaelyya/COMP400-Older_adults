import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional


class UserMemoryDB:
    def __init__(self):
        self.db_file = "user_memory.json"
        self.user_memories = self._load_memories()

    def _load_memories(self) -> Dict:
        """Load all user memories from file"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading memories: {e}")
            return {}

    def save_user_memory(self, user_id: str, user_info: Dict):
        """Save or update user memory"""
        try:
            # Add timestamp to track when the info was last updated
            user_info['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.user_memories[user_id] = user_info
            
            # Save to file
            with open(self.db_file, 'w') as f:
                json.dump(self.user_memories, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving user memory: {e}")

    def get_user_memory(self, user_id: str) -> Optional[Dict]:
        """Retrieve user memory if exists"""
        return self.user_memories.get(user_id)

    def has_user_memory(self, user_id: str) -> bool:
        """Check if user has existing memory"""
        return user_id in self.user_memories