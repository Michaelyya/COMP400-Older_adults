from typing import Dict, List, Optional
from dialogue_module import ExerciseDialogueSystem
from activity_module import ActivityAssistant
from vector_saving_module import ExerciseVectorDB
from user_module import UserMemoryDB
from openai import OpenAI
import json
import logging
from datetime import datetime
import os

class UnifiedAssistant:
    def __init__(self):
        self.exercise_system = ExerciseDialogueSystem()
        self.activity_assistant = ActivityAssistant()
        self.user_memory = UserMemoryDB()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.conversation_history: Dict[str, List[Dict]] = {}

    def _get_user_context(self, user_id: str) -> Dict:
        """Gather all available user context from various systems"""
        try:
            context = {}
            
            # Get exercise profile
            try:
                exercise_profile = self.exercise_system._load_user_profile(user_id)
                if exercise_profile:
                    context['exercise_profile'] = exercise_profile
            except Exception as e:
                logging.error(f"Error loading exercise profile: {e}")
                context['exercise_profile'] = {}

            # Get calendar data
            try:
                calendar = self.activity_assistant.calendar_advisor.get_calendar(user_id)
                if calendar:
                    context['calendar'] = calendar
            except Exception as e:
                logging.error(f"Error loading calendar: {e}")
                context['calendar'] = {}

            # Get user memory if available
            try:
                if hasattr(self.user_memory, 'get_user_memory'):
                    user_memory = self.user_memory.get_user_memory(user_id)
                    if user_memory:
                        context['user_memory'] = user_memory
            except Exception as e:
                logging.error(f"Error loading user memory: {e}")
                context['user_memory'] = {}

            return context

        except Exception as e:
            logging.error(f"Error getting user context: {e}")
            return {
                'exercise_profile': {},
                'calendar': {},
                'user_memory': {}
            }

    def _get_intent(self, message: str, user_context: Dict) -> Dict:
        """Determine message intent with better error handling"""
        try:
            # Check for common patterns first
            message_lower = message.lower()
            
            # Check for greetings
            if any(word in message_lower for word in ['hi', 'hello', 'hey']):
                return {"intent": "greeting", "requires_context": []}
            
            # Check for calendar-related keywords
            if any(word in message_lower for word in ['schedule', 'calendar', 'appointment', 'book', 'plan']):
                return {"intent": "calendar", "requires_context": ["calendar"]}
            
            # Check for exercise-related keywords
            if any(word in message_lower for word in ['exercise', 'workout', 'fitness', 'training']):
                return {"intent": "exercise", "requires_context": ["exercise"]}

            # If no clear pattern, use GPT for classification
            intent_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """Analyze the message and respond with one of these exact words:
                        'exercise' - for exercise recommendations
                        'calendar' - for schedule management
                        'integrated' - requires both exercise and calendar
                        'query' - general question about activities"""},
                    {"role": "user", "content": message}
                ],
                temperature=0.3
            )
            
            intent = intent_response.choices[0].message.content.strip().lower()
            return {
                "intent": intent,
                "requires_context": ["exercise", "calendar"] if intent == "integrated" else [intent]
            }
        except Exception as e:
            logging.error(f"Error determining intent: {e}")
            return {"intent": "query", "requires_context": []}

    def process_message(self, user_id: str, message: str) -> str:
        """Process user message with improved error handling"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        self.conversation_history[user_id].append({"role": "user", "content": message})
        
        try:
            # Get user context
            user_context = self._get_user_context(user_id)
            
            # First message or greeting
            if len(self.conversation_history[user_id]) == 1 or message.lower().strip() in ['hi', 'hello', 'hey']:
                greeting = """Hello! I'm here to help you with exercise recommendations and managing your schedule. 
                I can:
                - Suggest exercises based on your health needs
                - Manage your calendar and appointments
                - Help plan activities
                
                What would you like to know about?"""
                self.conversation_history[user_id].append({"role": "assistant", "content": greeting})
                return greeting
            
            # Determine intent
            intent_data = self._get_intent(message, user_context)
            
            # Process based on intent
            if intent_data["intent"] == "exercise":
                response = self.exercise_system.process_message(user_id, message)
                
            elif intent_data["intent"] == "calendar":
                response = self.activity_assistant.process_message(user_id, message)
                
            elif intent_data["intent"] == "integrated":
                response = self._handle_integrated_request(user_id, message, user_context)
                
            else:  # query or unknown
                response = self._handle_general_query(message, user_context)

            self.conversation_history[user_id].append({"role": "assistant", "content": response})
            return response

        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return """I'm here to help! Could you please let me know if you'd like to:
                   1. Get exercise recommendations
                   2. Check or update your schedule
                   3. Ask about your activities"""

    def _format_calendar_context(self, calendar_data: Dict) -> str:
        """Format calendar data with error handling"""
        try:
            if not calendar_data or 'schedule' not in calendar_data:
                return "No scheduled activities found."
                
            events = []
            for day in calendar_data['schedule']:
                for event in day.get('events', []):
                    events.append(f"{day['day']} at {event.get('time', 'TBD')}: {event.get('title', 'Unnamed event')}")
            
            return "Current schedule: " + "; ".join(events) if events else "No scheduled activities found."
        except Exception as e:
            logging.error(f"Error formatting calendar: {e}")
            return "Unable to retrieve schedule information."

    def _handle_integrated_request(self, user_id: str, message: str, context: Dict) -> str:
        """Handle integrated requests with better error handling"""
        try:
            # Get exercise recommendations
            exercise_response = self.exercise_system.process_message(user_id, message)
            
            # Try to schedule if appropriate
            if "recommend" in message.lower() or "suggest" in message.lower():
                calendar_response = self.activity_assistant.process_message(
                    user_id,
                    f"Schedule these activities: {exercise_response}"
                )
                return f"{exercise_response}\n\nI've also added these to your schedule:\n{calendar_response}"
            
            return exercise_response
            
        except Exception as e:
            logging.error(f"Error handling integrated request: {e}")
            return "I can help you with both exercises and scheduling. Which would you like to start with?"

    def _handle_general_query(self, message: str, context: Dict) -> str:
        """Handle general queries with improved response"""
        try:
            if any(word in message.lower() for word in ['hi', 'hello', 'hey']):
                return """Hello! I can help you with:
                       - Exercise recommendations
                       - Managing your schedule
                       - Planning activities
                       What would you like to know about?"""
                       
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Provide a helpful response about exercises or scheduling"},
                    {"role": "user", "content": message}
                ]
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error handling query: {e}")
            return "I can help you with exercises and scheduling. Which would you like to know more about?"

def main():
    assistant = UnifiedAssistant()
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Health & Activity Assistant ===")
    print("Welcome! I can help you with:")
    print("1. Exercise recommendations")
    print("2. Calendar management")
    print("3. Activity planning")
    print("\nType 'quit' to exit")
    print("================================\n")
    
    user_id = input("Please enter your name or ID: ").strip() or "user123"
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Take care! Goodbye!")
                break
            
            if user_input:
                response = assistant.process_message(user_id, user_input)
                print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye!")
            break
        except Exception as e:
            print("\nI'm here to help! What would you like to know about exercises or scheduling?")
            logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()