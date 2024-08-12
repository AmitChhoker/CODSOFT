import re

# Define a dictionary of rules and responses
RULES = {
    'hello|hi|hey': 'Hello! How can I assist you today?',
    'how are you': 'I\'m just a program, but I\'m doing great! How can I help you?',
    'what is your name': 'I am a bad Chatbot created by Amit for (CODSOFT). You can call me bad.',
    'bye|goodbye': 'Goodbye! Have a great day!',
    'default': 'Sorry, I didn\'t understand that. Can you please rephrase?'
}

# Function to get a response based on user input
def get_response(user_input):
    user_input = user_input.lower()
    
    for pattern, response in RULES.items():
        if re.search(pattern, user_input):
            return response
    
    return RULES['default']

# Main function to interact with the chatbot
def chat():
    print("Welcome to the ChatBot! Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ")
        
        if re.search(r'\bbye\b|\bgoodbye\b', user_input.lower()):
            print("ChatBot: " + get_response(user_input))
            break
        
        response = get_response(user_input)
        print("ChatBot: " + response)

if __name__ == "__main__":
    chat()
