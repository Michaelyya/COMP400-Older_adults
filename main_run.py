from flask import Flask, render_template, request, jsonify, session
from unified_assistant import UnifiedAssistant
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Initialize the unified assistant
assistant = UnifiedAssistant()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default_user')

        if not message:
            return jsonify({'response': 'Please enter a message'})

        # Process message through unified assistant
        response = assistant.process_message(user_id, message)
        
        # Log the interaction for debugging
        logger.info(f"User {user_id}: {message}")
        logger.info(f"Assistant response: {response}")
        
        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': "I apologize, but I'm having trouble with your request. Could you please try again?"
        })

@app.route('/get_profile', methods=['POST'])
def get_profile():
    """Get user profile and context"""
    try:
        data = request.json
        user_id = data.get('user_id', 'default_user')
        context = assistant._get_user_context(user_id)
        return jsonify(context)
    except Exception as e:
        logger.error(f"Error getting profile: {str(e)}")
        return jsonify({'error': 'Failed to get user profile'})

def create_app():
    """Application factory function"""
    # Ensure required directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('user_profiles', exist_ok=True)
    
    return app

if __name__ == '__main__':
    create_app().run(debug=True, port=5000)