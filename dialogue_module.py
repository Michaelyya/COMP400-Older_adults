from typing import Dict, List, Optional
from openai import OpenAI
import json
import logging
from datetime import datetime
import os
from dataclasses import dataclass
from vector_saving_module import ExerciseVectorDB

logging.getLogger('httpx').setLevel(logging.WARNING)  # Add this line to suppress HTTP logs
logging.basicConfig(level=logging.INFO)
@dataclass
class UserProfile:
    user_id: str
    age: Optional[int] = None
    health_conditions: List[str] = None
    mobility_level: Optional[str] = None
    exercise_preferences: List[str] = None
    exercise_history: Optional[str] = None
    safety_concerns: List[str] = None
    preferred_intensity: Optional[str] = None
    preferred_location: Optional[str] = None
    available_times: List[str] = None

class ExerciseDialogueSystem:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.vector_db = ExerciseVectorDB()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.conversation_history: Dict[str, List[Dict]] = {}
        
        # Load exercise database
        self._initialize_exercise_db()

    def _initialize_exercise_db(self):
        """Initialize the exercise vector database"""
        if not self.vector_db.load_index():
            self.vector_db.load_exercises('old_adults RAG/exercises/exercise.json')
            self.vector_db.create_embeddings()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the conversation"""
        return """You are a knowledgeable and empathetic exercise companion for older adults. 
        Your goal is to gather information about their health, preferences, and concerns to recommend suitable exercises.
        
        Follow these guidelines:
        1. Be patient and supportive
        2. Ask one question at a time
        3. Acknowledge their concerns
        4. Use simple, clear language
        5. Focus on safety and comfort
        6. Show enthusiasm for their interest in exercise
        
        Required information to collect:
        - Age
        - Health conditions
        - Mobility level
        - Exercise preferences
        - Safety concerns
        - Preferred exercise intensity
        - Preferred location (home, gym, outdoors)
        - Available times
        
        Once you have gathered sufficient information, summarize it and ask for confirmation."""

    def _extract_user_info(self, conversation: List[Dict]) -> Dict:
        """Extract user information from conversation"""
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """Extract user information from the conversation and format as JSON. 
                    Include these fields if mentioned:
                    - age
                    - health_conditions (list)
                    - mobility_level
                    - exercise_preferences (list)
                    - safety_concerns (list)
                    - preferred_intensity
                    - preferred_location
                    - available_times (list)
                    
                    Example format:
                    {
                        "age": 70,
                        "health_conditions": ["arthritis", "balance issues"],
                        "mobility_level": "moderate",
                        "exercise_preferences": ["walking", "gentle exercises"],
                        "safety_concerns": ["falling", "joint pain"],
                        "preferred_intensity": "low",
                        "preferred_location": "home and park",
                        "available_times": ["mornings"]
                    }
                    """},
                    {"role": "user", "content": conversation_text}
                ],
                temperature=0.3  # Lower temperature for more consistent formatting
            )
            
            # Extract JSON from the response text
            response_text = response.choices[0].message.content
            # Find JSON content between curly braces
            json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Error extracting user information: {e}")
            return {}

    def _save_user_profile(self, user_id: str, profile: Dict):
        """Save user profile to JSON file"""
        file_path = f"user_profiles/{user_id}.json"
        os.makedirs("user_profiles", exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(profile, f, indent=2)

    def _load_user_profile(self, user_id: str) -> Optional[Dict]:
        """Load user profile from JSON file"""
        file_path = f"user_profiles/{user_id}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

    def process_message(self, user_id: str, message: str) -> str:
        """Process user message and generate response"""
        # Initialize conversation history if needed
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        # Add user message to history
        self.conversation_history[user_id].append({"role": "user", "content": message})

        try:
            # Generate response using GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    *self.conversation_history[user_id]
                ],
                temperature=0.7
            )

            assistant_response = response.choices[0].message.content
            self.conversation_history[user_id].append({"role": "assistant", "content": assistant_response})

            # Check if we have gathered enough information
            user_info = self._extract_user_info(self.conversation_history[user_id])
            
            # If we have sufficient information, save profile and get exercise recommendations
            if self._is_profile_complete(user_info):
                self._save_user_profile(user_id, user_info)
                recommendations = self._get_exercise_recommendations(user_info)
                
                # Format recommendations as a friendly message
                recommendation_response = self._format_recommendations(recommendations)
                
                # Add recommendations to conversation
                self.conversation_history[user_id].append(
                    {"role": "assistant", "content": recommendation_response}
                )
                return recommendation_response

            return assistant_response

        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return "I apologize, but I'm having trouble processing your request. Could you please try again?"

    def _is_profile_complete(self, profile: Dict) -> bool:
        """Check if we have gathered enough information to make recommendations"""
        required_fields = ['health_conditions', 'preferred_intensity']
        return len(profile.get('health_conditions', [])) > 0 and 'preferred_intensity' in profile

    def _get_exercise_recommendations(self, user_info: Dict) -> List[Dict]:
        """Get exercise recommendations based on user profile"""
        # Construct search query based on user information
        query = f"""
        Looking for exercises suitable for someone with:
        Health conditions: {', '.join(user_info['health_conditions'])}
        Mobility level: {user_info['mobility_level']}
        Preferred intensity: {user_info['preferred_intensity']}
        Exercise preferences: {', '.join(user_info.get('exercise_preferences', []))}
        """
        
        # Get recommendations from vector database
        return self.vector_db.search_exercises(query, num_results=3)

    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """Format exercise recommendations as a friendly message"""
        response = "Based on what you've told me, here are some exercises that might be good for you:\n\n"
        
        for i, exercise in enumerate(recommendations, 1):
            response += f"{i}. {exercise.name}\n"
            response += f"   Difficulty: {exercise.difficulty}\n"
            response += f"   Description: {exercise.description}\n"
            response += f"   Benefits: {', '.join(exercise.benefits)}\n"
            response += f"   Safety tips: {', '.join(exercise.original_data['safety_tips'])}\n\n"

        response += "\nThank you for sharing your information with me. These recommendations are based on your specific needs and conditions. Remember to start slowly and listen to your body. If you experience any pain or discomfort, please stop and consult with a healthcare professional."
        return response
def main():
    # Initialize the dialogue system
    dialogue_system = ExerciseDialogueSystem()
    
    # Welcome message
    print("\n=== Exercise Recommendation System ===")
    print("Welcome! I'm here to help you find suitable exercises.")
    print("I'll ask you some questions to understand your needs better.")
    print("Type 'quit' to exit at any time.")
    print("=======================================\n")
    
    # Get user ID or generate one
    user_id = input("Please enter your name or ID: ").strip() or "user123"
    print("\nExercise Assistant: Hello! Let's find some suitable exercises for you. Could you tell me about your age and any health conditions you might have?")
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for quit command
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nExercise Assistant: Thank you for chatting! Take care and stay active!")
                break
            
            # Process message and get response
            if user_input:  # Only process non-empty messages
                response = dialogue_system.process_message(user_id, user_input)
                print(f"\nExercise Assistant: {response}")
                
                # Check if response contains exercise recommendations
                if "Based on what you've told me, here are some exercises" in response:
                    print("\n=== End of Conversation ===")
                    print("Exercise recommendations have been provided. Take care and stay active!")
                    break

        except KeyboardInterrupt:
            print("\n\nExercise Assistant: Goodbye! Take care!")
            break
            
        except Exception as e:
            print(f"\nSorry, there was an error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    # Configure logging
    logging.getLogger('httpx').setLevel(logging.WARNING)  # Suppress HTTP logs
    logging.basicConfig(level=logging.ERROR)  # Only show errors
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")