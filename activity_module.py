from typing import Dict, Optional, List
from dialogue_module import ExerciseDialogueSystem
from openai import OpenAI
import json
import logging
from datetime import datetime
import os
from dataclasses import dataclass
from vector_saving_module import ExerciseVectorDB
from schedule_module import Calendaradvisor

class ActivityAssistant:
    def __init__(self):
        self.exercise_system = ExerciseDialogueSystem()
        self.calendar_advisor = Calendaradvisor()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.conversation_history: Dict[str, List[Dict]] = {}

    def _get_system_prompt(self) -> str:
        return """You are an intelligent activity assistant for older adults, helping them manage both their exercise routines and daily schedule. 

        Your capabilities include:
        1. Understanding exercise needs and providing recommendations
        2. Managing calendar events and appointments
        3. Scheduling exercise sessions at appropriate times
        4. Helping maintain a balanced daily routine

        When handling calendar requests:
        1. Extract the time, day, and activity details
        2. Format events in 24-hour time format
        3. Include relevant notes for the elderly user
        4. Specify duration if mentioned
        5. Categorize the activity type appropriately

        For calendar updates, return a complete JSON structure with:
        - user_id
        - schedule (array of days and events)
        - last_updated timestamp
        - action_taken description"""

    def process_message(self, user_id: str, message: str) -> str:
        """Process user message and determine appropriate response type"""
        
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        self.conversation_history[user_id].append({"role": "user", "content": message})

        try:
            # First, determine if this is a calendar-related request
            intent_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """Classify this message into one of these categories:
                        - calendar_add (adding new events)
                        - calendar_view (viewing schedule)
                        - calendar_delete (removing events)
                        - exercise (exercise related)
                        Respond with only the classification."""},
                    {"role": "user", "content": message}
                ],
                temperature=0.3
            )

            intent = intent_response.choices[0].message.content.strip().lower()

            if 'calendar' in intent:
                # For calendar operations, format the request appropriately
                calendar_prompt = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": """Format this calendar request into a structured command.
                            For adding: "add event: [time] on [day] - [activity] (duration: [X] minutes)"
                            For viewing: "view calendar"
                            For deleting: "delete event: [time] on [day] - [activity]"
                            Use 24-hour time format."""},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.3
                )
                
                formatted_request = calendar_prompt.choices[0].message.content.strip()
                
                if intent == 'calendar_add':
                    calendar_response = self.calendar_advisor.process_calendar_query(formatted_request, user_id)
                    response = self._format_calendar_response(calendar_response)
                    if 'error' in calendar_response:
                        response = "I apologize, but I couldn't add that to your calendar. Could you please try rephrasing your request?"
                
                elif intent == 'calendar_view':
                    calendar = self.calendar_advisor.get_calendar(user_id)
                    response = self._format_calendar_response(calendar)
                
                elif intent == 'calendar_delete':
                    calendar_response = self.calendar_advisor.process_calendar_query(formatted_request, user_id)
                    response = self._format_calendar_response(calendar_response)
                    if 'error' in calendar_response:
                        response = "I apologize, but I couldn't delete that event. Could you please specify which event you'd like to delete?"
            
            else:  # Exercise-related query
                response = self.exercise_system.process_message(user_id, message)

            self.conversation_history[user_id].append({"role": "assistant", "content": response})
            return response

        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return "I apologize, but I'm having trouble understanding your request. Could you please rephrase it? For calendar events, try specifying the time, day, and activity clearly."

    def _format_calendar_response(self, calendar_data: Dict) -> str:
        """Format calendar data into readable text"""
        if not calendar_data or 'schedule' not in calendar_data:
            return "Your calendar is currently empty. You can add events by specifying the time, day, and activity."
            
        response = "📅 Here's your schedule:\n\n"
        
        for day in calendar_data['schedule']:
            response += f"🗓️ {day['day']}:\n"
            if day['events']:
                for event in sorted(day['events'], key=lambda x: x['time']):
                    # Convert 24-hour time to 12-hour format for display
                    time_24h = event['time']
                    try:
                        time_obj = datetime.strptime(time_24h, '%H:%M')
                        time_12h = time_obj.strftime('%I:%M %p')
                        response += f"   • {time_12h}: {event['title']}"
                        if 'duration' in event:
                            response += f" ({event['duration']} mins)"
                        if 'notes' in event:
                            response += f"\n     📝 Note: {event['notes']}"
                        response += "\n"
                    except ValueError:
                        response += f"   • {time_24h}: {event['title']}\n"
            else:
                response += "   No events scheduled\n"
            response += "\n"
        
        if 'action_taken' in calendar_data:
            response += f"\n✅ {calendar_data['action_taken']}"
            
        return response

    def add_calendar_event(self, user_id: str, day: str, time: str, title: str, duration: Optional[str] = None, notes: Optional[str] = None) -> str:
        """Helper method to add a calendar event"""
        query = f"add event: {time} on {day} - {title}"
        if duration:
            query += f" (duration: {duration} minutes)"
        if notes:
            query += f" note: {notes}"
        
        calendar_response = self.calendar_advisor.process_calendar_query(query, user_id)
        return self._format_calendar_response(calendar_response)

def main():
    assistant = ActivityAssistant()
    
    print("\n=== Activity Assistant for Older Adults ===")
    print("Welcome! I can help you with:")
    print("1. Managing your calendar (add, view, or delete events)")
    print("2. Exercise recommendations")
    print("\nTip: For calendar events, please specify:")
    print("- The day (e.g., Monday, Tuesday)")
    print("- The time (e.g., 2 PM, 14:00)")
    print("- The activity (e.g., visit parents, doctor appointment)")
    print("\nType 'quit' to exit")
    print("=====================================\n")
    
    user_id = input("Please enter your name or ID: ").strip() or "user123"
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Goodbye! Take care!")
                break
            
            if user_input:
                response = assistant.process_message(user_id, user_input)
                print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye! Take care!")
            break
            
        except Exception as e:
            print(f"\nSorry, there was an error. Please try again with a clearer request.")
            logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()