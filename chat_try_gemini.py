import streamlit as st
import re
import random
import nltk
import os
import google.generativeai as genai
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from dotenv import load_dotenv
import emoji

import os

# Get the current directory of the .py file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the img folder
img_folder = os.path.join(current_dir, "img")

hitesh_image = os.path.join(img_folder, "hitesh.jpeg")
piyush_image = os.path.join(img_folder, "piyush.jpg")
donald_image = os.path.join(img_folder, "donald.jpg")



# Constants
HINDI_EXPRESSION_PROBABILITY = 0.2
TECH_EMPHASIS_PROBABILITY = 0.15
FORMAL_REPLACEMENT_PROBABILITY = 0.3
EMOJI_USAGE_THRESHOLD = 0.3
QUESTION_RATIO_THRESHOLD = 0.5
FORMAL_LANGUAGE_THRESHOLD = 0.5
LONG_CONTENT_THRESHOLD = 300
SHORT_QUERY_THRESHOLD = 3
CONTEXT_MESSAGES_LIMIT = 10

# Load environment variables from .env file
load_dotenv()

# Download necessary NLTK data (only runs once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# --- Helper Functions for Hitesh Style ---
HINDI_EXPRESSIONS = [
    "Bilkul", "Ekdum sahi", "Bahut badhiya", "Aap samajh rahe ho?", "Dekho",
    "Matlab", "Aur ye important hai", "Ye samajh lo", "Dhyaan se suno"
]

def add_hindi_expression(segment):
    available_expressions = [e for e in HINDI_EXPRESSIONS if e not in st.session_state.style_elements_used]
    if available_expressions:
        expression = random.choice(available_expressions)
        st.session_state.style_elements_used.add(expression)
        return f"{expression}! {segment}"
    return segment  # Return original segment if no expressions available

def add_tech_emphasis(segment):
    tech_emphasis = [
        "Ye concept industry mein bahut use hota hai. ",
        "Real projects mein aise hi kaam hota hai. ",
        "Interview mein ye zaroor puchha jata hai. "
    ]
    return f"{segment} {random.choice(tech_emphasis)}"


def display_message(role, content, image_path=None):
    """Displays a chat message with optional image."""
    with st.chat_message(role):
        if image_path:
            st.image(image_path, width=50)  # Adjust width as needed
        st.markdown(content)

def main():
    st.set_page_config(page_title="Adaptive Chat with Google Gemini", layout="wide")
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "option_selected" not in st.session_state:
        st.session_state.option_selected = None
    
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "topics": Counter(),
            "sentiment_history": [],
            "question_frequency": 0,
            "avg_message_length": 0,
            "message_count": 0,
            "technical_terms": Counter(),
            "formal_language": 0,
            "emoji_usage": 0,
        }
    
    if "last_greeting_index" not in st.session_state:
        st.session_state.last_greeting_index = -1
        
    if "style_elements_used" not in st.session_state:
        st.session_state.style_elements_used = set()
    
    # Load API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
        st.stop()
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Google Gemini Configuration")
        st.success("✅ API Key loaded from environment variable")
        
        st.divider()
        
        model = st.selectbox("Select Gemini Model", ["gemini-1.5-pro", "gemini-1.5-flash"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        st.divider()
        
        if st.session_state.user_profile["message_count"] > 0:
            st.subheader("User Profile Metrics")
            metrics = get_user_metrics()  # Assumes this is defined elsewhere
            st.write(f"Avg Sentiment: {metrics['avg_sentiment']:.2f}")
            st.write(f"Technical Language: {metrics['technical_ratio']:.2f}")
            st.write(f"Formality Level: {metrics['formal_language']:.2f}")
            st.write(f"Question Ratio: {metrics['question_ratio']:.2f}")
            st.write(f"Uses Emojis: {'Yes' if metrics['emoji_usage'] > 0.3 else 'No'}")
            if metrics['top_topics']:
                st.write(f"Top Topics: {', '.join(metrics['top_topics'][:3])}")

        # 🔄 Reset button in sidebar
        if st.button("🔄 Reset Chat"):
            reset_chat()
            st.rerun()

    # Mentor profiles
    profiles = [
        {"image": hitesh_image, "name": "Hitesh Choudhary", "option": "Option 1: Hitesh Style"},
        {"image": piyush_image, "name": "Piyush Garg", "option": "Option 2: Piyush Style"},
    ]
    
    def get_mentor_name(option):
        if "Option 1" in option:
            return "Hitesh Choudhary"
        elif "Option 2" in option:
            return "Piyush Garg"

    if st.session_state.option_selected:
        mentor_name = get_mentor_name(st.session_state.option_selected)
        st.title(f"Chat with {mentor_name}")
        
        # 👉 Reset button in top-right corner
        top_col1, top_col2, top_col3 = st.columns([6, 1, 1])
        with top_col3:
            if st.button("🔄 Reset"):
                reset_chat()
                st.rerun()
        
        mentor_index = 0 if "Option 1" in st.session_state.option_selected else 1
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(profiles[mentor_index]["image"], width=100)
        with col2:
            st.write(f"## You're chatting with {profiles[mentor_index]['name']}")
        st.divider()
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("What would you like to talk about?"):
            analyze_user_input(prompt)  # Assumes this function is defined
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                with st.spinner("Getting response from Google Gemini..."):
                    raw_content = get_gemini_response(prompt, model, temperature)  # Assumes this function is defined
                with st.chat_message("assistant"):
                    if "Option 1" in st.session_state.option_selected:
                        response = transform_to_hitesh_style(prompt, raw_content)
                    elif "Option 2" in st.session_state.option_selected:
                        response = transform_to_piyush_style(prompt, raw_content)
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error communicating with Google Gemini: {str(e)}")

    else:
        st.title("Chat with Chai with Code Mentors")
        st.write("Welcome! Please select a mentor below:")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(profiles[0]["image"], width=200)
            st.write(f"### {profiles[0]['name']}")
            if st.button("Select Hitesh", key="btn_hitesh"):
                st.session_state.option_selected = profiles[0]["option"]
                reset_chat(preserve_option=True)
                st.rerun()

        with col2:
            st.image(profiles[1]["image"], width=200)
            st.write(f"### {profiles[1]['name']}")
            if st.button("Select Piyush", key="btn_piyush"):
                st.session_state.option_selected = profiles[1]["option"]
                reset_chat(preserve_option=True)
                st.rerun()

        st.divider()
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# 🔧 Utility function to reset chat
def reset_chat(preserve_option=False):
    if not preserve_option:
        st.session_state.option_selected = None
    st.session_state.messages = []
    st.session_state.last_greeting_index = -1
    st.session_state.style_elements_used = set()
    st.session_state.user_profile = {
        "topics": Counter(),
        "sentiment_history": [],
        "question_frequency": 0,
        "avg_message_length": 0,
        "message_count": 0,
        "technical_terms": Counter(),
        "formal_language": 0,
        "emoji_usage": 0,
    }



def get_gemini_response(prompt, model_name, temperature):
    """Get a response from the Google Gemini API"""
    try:
        # Create conversation context from past messages (last 10 messages)
        context_messages = []
        if len(st.session_state.messages) > 0:
            context_messages = st.session_state.messages[-min(10, len(st.session_state.messages)):]
        
        # Prepare the conversation history in Gemini's format
        history = []
        for msg in context_messages:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature}
        )
        
        # Create a chat session if there's history, otherwise generate content directly
        if history:
            chat = model.start_chat(history=history)
            response = chat.send_message(prompt)
        else:
            response = model.generate_content(prompt)
        
        # Return the text content of the response
        return response.text
    
    except Exception as e:
        st.error(f"Google Gemini API Error: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please check your API setup and try again."

def analyze_user_input(text):
    """Analyze user input to build a profile of their communication style"""
    profile = st.session_state.user_profile
    
    # Update message count and length metrics
    profile["message_count"] += 1
    total_length = profile["avg_message_length"] * (profile["message_count"] - 1) + len(text)
    profile["avg_message_length"] = total_length / profile["message_count"]
    
    # Check for questions
    if '?' in text:
        profile["question_frequency"] += 1
    
    # Analyze sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    profile["sentiment_history"].append(sentiment_scores["compound"])
    
    # Identify potential topics (simple keyword extraction)
    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        if len(word) > 3 and word not in ["this", "that", "what", "when", "where", "which", "there", "their"]:
            profile["topics"][word] += 1
    
    # Detect technical terms
    technical_keywords = ["javascript", "python", "code", "programming", "developer", "web", "api", 
                         "function", "variable", "class", "framework", "library", "database", "server"]
    for term in technical_keywords:
        if term in text.lower():
            profile["technical_terms"][term] += 1
    
    # Check for formal language markers
    formal_markers = ["would", "could", "should", "shall", "may", "might", "nevertheless", 
                     "however", "therefore", "thus", "hence", "consequently", "furthermore"]
    formal_count = sum(1 for word in words if word in formal_markers)
    profile["formal_language"] = ((profile["formal_language"] * (profile["message_count"] - 1)) + 
                                 (formal_count > 0)) / profile["message_count"]
    
    # Check for emoji usage
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               "]+", flags=re.UNICODE)
    emojis = emoji_pattern.findall(text)
    profile["emoji_usage"] = ((profile["emoji_usage"] * (profile["message_count"] - 1)) + 
                             (len(emojis) > 0)) / profile["message_count"]

def get_user_metrics():
    """Extract useful metrics from user profile for response generation"""
    profile = st.session_state.user_profile
    
    # Calculate average sentiment (-1.0 to 1.0)
    avg_sentiment = sum(profile["sentiment_history"]) / len(profile["sentiment_history"]) if profile["sentiment_history"] else 0
    
    # Get top topics
    top_topics = [topic for topic, count in profile["topics"].most_common(3)]
    
    # Calculate question ratio
    question_ratio = profile["question_frequency"] / profile["message_count"] if profile["message_count"] > 0 else 0
    
    # Determine if user uses technical language
    technical_ratio = sum(profile["technical_terms"].values()) / profile["message_count"] if profile["message_count"] > 0 else 0
    
    return {
        "avg_sentiment": avg_sentiment,
        "top_topics": top_topics,
        "question_ratio": question_ratio,
        "message_length": profile["avg_message_length"],
        "technical_ratio": technical_ratio,
        "formal_language": profile["formal_language"],
        "emoji_usage": profile["emoji_usage"]
    }

def transform_to_hitesh_style(prompt, content):
    """Transform content to Hitesh's natural Hinglish speaking style"""
    # Detect basic context
    is_greeting = any(word in prompt.lower() for word in ["hello", "hi", "hey", "namaste"])
    is_tech_question = any(word in prompt.lower() for word in ["javascript", "js", "react", "code", "coding", "web", "html", "css", "programming"])
    is_short_query = len(prompt.split()) <= SHORT_QUERY_THRESHOLD

    # Handle greetings with natural Hinglish responses
    if is_greeting:
        greeting_responses = [
            "Arrey bhai sahab! Namaste! Kaise ho aap? How can I help you today?",
            "Hello bhai! Kya haal hai? Tell me, mai kaise help kar sakta hoon aapko?",
            "Are waah! Aap aa gaye! Batao, kya help chahiye aapko?",
            "Namaste dost! Chai pe charcha karte hain aaj. What's on your mind today?",
            "Hello hello! Kya chal raha hai aaj? Tell me what you're interested in learning!"
        ]
        
        # Avoid using the same greeting twice in a row
        available_indexes = list(range(len(greeting_responses)))
        if hasattr(st.session_state, 'last_greeting_index') and st.session_state.last_greeting_index in available_indexes:
            available_indexes.remove(st.session_state.last_greeting_index)
        
        greeting_index = random.choice(available_indexes)
        st.session_state.last_greeting_index = greeting_index
        
        return greeting_responses[greeting_index]
    
    if re.search(r'\bwho are you\b|\bwhat is your name\b|\bwho am i talking to\b', prompt.lower()):
        return "Hello! I'm Hitesh Choudhary, a tech educator and content creator focused on web development and programming. I run the YouTube channel 'Chai aur Code' where I teach programming concepts in a simple, friendly style mixing Hindi and English. Mera goal hai coding ko simple banana, taki sab log seekh sake! How can I help you today?"
    
    # For short queries without tech context, keep response simple but still conversational
    if is_short_query and not is_tech_question:
        # Add a bit of Hinglish flair even to short responses
        prefixes = ["Hanji, ", "Haan, ", "Bilkul, ", "Dekhiye, ", "Acha, ", ""]
        return random.choice(prefixes) + content
    
    # Split content for natural manipulation
    paragraphs = content.split('\n\n')
    main_content = paragraphs[0] if paragraphs else content
    sentences = re.split(r'(?<=[.!?])\s+', main_content)
    
    # Tech-specific intros in authentic Hinglish
    if is_tech_question:
        # Detect which tech is mentioned
        tech_terms = {
            "javascript": [
                "JavaScript ke baare mein baat karte hain. Dekho, javascript actually ek bahut hi powerful language hai. ",
                "Aaj hum javascript par focus karenge. Main personally javascript ko bahut pasand karta hoon kyunki ye versatile hai. ",
                "JavaScript! Mere favorite topic mein se ek. Is language ko samajhna bahut zaroori hai aaj ke time mein. "
            ],
            "js": [
                "JS ke baare mein baat karte hain. Dekho, javascript actually ek bahut hi powerful language hai. ",
                "Aaj hum JS par focus karenge. Main personally JS ko bahut pasand karta hoon kyunki ye versatile hai. ",
                "JS! Mere favorite topic mein se ek. Is language ko samajhna bahut zaroori hai aaj ke time mein. "
            ],
            "react": [
                "React! Mera favorite framework hai ye. Dekho, ismein components ka concept hai jo life bahut aasan bana deta hai. ",
                "Chalo aaj React ke baare mein baat karte hain. Main personally React se bahut saare projects banata hoon. ",
                "React ka concept samajhte hain aaj. Dekho frontend development mein React ne revolution la diya hai. "
            ],
            "web": [
                "Web development ki baat karte hain. Dekho, ismein teen cheezein important hain - HTML, CSS aur JavaScript. ",
                "Web development! Meri expertise hai ye. Main batata hoon ki isme kya important hai. ",
                "Web development ke liye mindset zaroori hai. Concepts clear hone chahiye, baaki sab aata rahega. "
            ],
            "html": [
                "HTML yaani ki Hypertext Markup Language. Website ka skeleton hai ye! Isey samjhe bina aage nahi badh sakte. ",
                "HTML ke baare mein baat karte hain. Dekho, ye web development ki building block hai. ",
                "HTML simple hai lekin powerful bhi. Main personally har project mein semantic HTML use karta hoon. "
            ],
            "css": [
                "CSS yaani ki Cascading Style Sheets. Iske bina website boring lagegi! Main batata hoon kaise use karna hai. ",
                "CSS pe baat karte hain aaj. Flexbox aur Grid ab standard ban chuke hain industry mein. ",
                "CSS ka concept simple hai - styling karna! Lekin ismein mastery hasil karna thoda challenging hai. "
            ],
            "python": [
                "Python ek bahut hi simple language hai. Main khud ise bahut use karta hoon automations ke liye. ",
                "Python ke baare mein baat karte hain. Ye data science, web development, har jagah use hoti hai. ",
                "Python! Beginners ke liye perfect language. Syntax simple hai aur community bhi bahut helpful hai. "
            ]
        }
        
        # Find which tech term was mentioned and use appropriate intro
        mentioned_tech = next((term for term in tech_terms.keys() if term in prompt.lower()), None)
        
        if mentioned_tech:
            intro = random.choice(tech_terms[mentioned_tech])
        else:
            # General tech intro with natural Hinglish
            general_tech_intros = [
                "Dekho, programming mein sabse important cheez hai concepts clear hona. Mai hamesha kehta hoon practice karo, code likho, projects banao. ",
                "Tech industry mein aage badhna hai toh fundamentals strong hone chahiye. Main personally har roz code likhta hoon, aap bhi try karo. ",
                "Coding seekhna hai toh consistency zaroori hai. Main apne experience se bata raha hoon, daily thoda thoda practice karo. "
            ]
            intro = random.choice(general_tech_intros)
    else:
        # Non-tech questions get more casual, conversational intros
        casual_intros = [
            "Acha question hai ye! Main apne hisaab se batata hoon. ",
            "Ye bahut interesting topic hai! Mere experience se bataun toh ",
            "Haan, is baare mein baat karte hain. Maine bhi iske baare mein socha hai. ",
            "Dekho, main isme expert nahi hoon, but jo mujhe pata hai wo share karta hoon. ",
            "Ye question bahut baar pucha jaata hai! Main apna perspective share karta hoon. "
        ]
        intro = random.choice(casual_intros)
    
    # Process content to sound more natural in Hinglish
    transformed_sentences = []
    transformed_sentences.append(intro)
    
    # Pattern for inserting Hinglish phrases naturally
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        # Skip very short sentences
        if len(sentence.split()) < 3:
            transformed_sentences.append(sentence)
            continue
        
        # Naturally insert Hindi expressions based on sentence position and content
        if i == 0 or i % 3 == 0:  # Every few sentences add Hindi flavor
            hinglish_patterns = [
                # Agreement patterns
                (r'\byes\b|correct|right|true', ["Bilkul sahi! ", "Haan ekdum! ", "Sahi baat hai! "]),
                # Important point patterns
                (r'\bimportant|crucial|key|essential', ["Ye bahut zaroori hai! ", "Dhyan rakhna ye important hai! ", "Ye point note kar lo! "]),
                # Explanation patterns
                (r'\bfor example|instance|such as', ["Jaise ki ", "Matlab ", "For example, "]),
                # Conclusion patterns
                (r'\bTherefore|Thus|So|Finally', ["Toh ismein ", "Iska matlab ", "Toh finally "]),
            ]
            
            modified = False
            for pattern, replacements in hinglish_patterns:
                if re.search(pattern, sentence, re.IGNORECASE) and not modified:
                    if random.random() < 0.7:  # 70% chance to apply pattern
                        sentence = random.choice(replacements) + sentence
                        modified = True
            
            # If no patterns matched but we still want Hindi flavor
            if not modified and random.random() < 0.3:
                hindi_fillers = [
                    "Dekho, ",
                    "Samjho, ",
                    "Simple hai, ",
                    "Asaan bhasha mein, ",
                    "Main batata hoon, "
                ]
                sentence = random.choice(hindi_fillers) + sentence
        
        transformed_sentences.append(sentence)
    
    # Join the transformed sentences
    result = " ".join(transformed_sentences)
    
    # Add remaining paragraphs
    if len(paragraphs) > 1:
        additional_content = '\n\n'.join(paragraphs[1:])
        result += f"\n\n{additional_content}"
    
    # Add authentic Hinglish conclusions for larger responses
    if len(result) > 300 and random.random() < 0.8:
        conclusions = [
            "\n\nToh dosto, yahi hai mere thoughts is topic pe. Practice karte raho, questions puchte raho!",
            "\n\nAur yaad rakhiye, coding sikhne ke liye consistency zaroori hai. Thoda thoda karke seekho, lekin daily karo.",
            "\n\nAb samajh mein aaya hoga aapko! Koi doubt ho toh puch lena, main explain kar dunga phir se.",
            "\n\nBas yehi kehna chahta hoon main - concepts clear karo, practice karo, aur enjoy karo journey ko!",
            "\n\nChalo dosto, aaj ke liye itna hi. Hope ye explanation helpful raha hoga. Keep coding and stay curious!",
            "\n\nMaine apne experience se jo seekha hai, wahi share kiya hai. Ab aage aap khud explore karo aur apna perspective banaao."
        ]
        result += random.choice(conclusions)
    
    # Add tech-specific motivation for programming questions
    if is_tech_question and random.random() < 0.6:
        tech_motivations = [
            "\n\nYaad rakhna, har developer struggle karta hai shuru mein. Maine bhi kiya tha. Lekin consistent practice se sab aata hai!",
            "\n\nMain personally kehta hoon ki tutorials se zyada projects pe kaam karo. Real learning projects se hi hoti hai.",
            "\n\nDekho, tech industry mein ek cheez guarantee hai - change! Toh learning kabhi band mat karna, yehi success ka mantra hai.",
            "\n\nAur haan, Stack Overflow aur documentation aapke best friends hain. Use them wisely!"
        ]
        result += random.choice(tech_motivations)
    
    # Add chai reference occasionally
    if random.random() < 0.25:
        chai_references = [
            "\n\nChalo, ek cup chai peete hain aur code karte hain! ☕",
            "\n\nAb thodi chai pee lo, dimag refresh karo, aur phir practice karna shuru karo! ☕",
            "\n\nYe concepts samajhne ke liye chai zaroori hai! ☕ Chai with Code, yaad rakhna!",
            "\n\nChai piyo, code karo, developer bano! ☕"
        ]
        result += random.choice(chai_references)
    
    # Occasionally add signature emoji but not too many
    if "☕" not in result and random.random() < 0.3:
        emojis = ["🔥", "💻", "👨‍💻", "✅", "🚀", "💪"]
        result = result.rstrip() + f" {random.choice(emojis)}"
    
    return result

def transform_to_piyush_style(prompt, content):
    """Transform content to Piyush Garg's natural Hinglish speaking style"""
    # Detect basic context
    is_greeting = any(word in prompt.lower() for word in ["hello", "hi", "hey", "namaste"])
    is_tech_question = any(word in prompt.lower() for word in ["javascript", "js", "react", "node", "coding", "web", "programming"])
    is_short_query = len(prompt.split()) <= SHORT_QUERY_THRESHOLD
    
    # Handle greetings with Piyush's style
    if is_greeting:
        greeting_responses = [
            "Hello everybody! Kaise ho aap log? How can I help you today?",
            "Hello bhai log! Hope sab badhiya hai! Tell me, how can I assist you?",
            "Hello everyone! Aaj hum kya discuss karne wale hain?",
            "Hello hello! Aap ka swagat hai! What's on your mind today?",
            "Hello friends! Chaliye kuch interesting discuss karte hain aaj."
        ]
        
        # Avoid using the same greeting twice
        available_indexes = list(range(len(greeting_responses)))
        if hasattr(st.session_state, 'last_greeting_index') and st.session_state.last_greeting_index in available_indexes:
            available_indexes.remove(st.session_state.last_greeting_index)
        
        greeting_index = random.choice(available_indexes)
        st.session_state.last_greeting_index = greeting_index
        
        return greeting_responses[greeting_index]
    
    if re.search(r'\bwho are you\b|\bwhat is your name\b|\bwho am i talking to\b', prompt.lower()):
        return "Hello everyone! I'm Piyush Garg, a software developer and tech educator. I create programming tutorials and tech content on my YouTube channel where I explain complex concepts in a simple way. Mai hamesha kehta hoon fundamentals strong rakho! Let me know what you'd like to learn about today!"
    
    # For short queries without tech context, keep response simple but with Piyush's style
    if is_short_query and not is_tech_question:
        # Add a bit of Piyush's flair even to short responses
        prefixes = ["Haan bhai, ", "Absolutely, ", "Dekhiye, ", "Haan, ", ""]
        return random.choice(prefixes) + content
    
    # Split content for manipulation
    paragraphs = content.split('\n\n')
    main_content = paragraphs[0] if paragraphs else content
    sentences = re.split(r'(?<=[.!?])\s+', main_content)
    
    # Tech-specific intros in Piyush's style
    if is_tech_question:
        # Detect which tech is mentioned
        tech_terms = {
            "javascript": [
                "Javascript ke baare mein baat karte hain. Ye ek beautiful language hai, aur ye browser environment mein execute hoti hai. ",
                "Let's talk about Javascript today. Mai personally javascript mein bahut saare projects karta hoon. ",
                "Javascript ek amazing language hai. Ismein aap browser mein bhi code likh sakte ho aur backend mein bhi. "
            ],
            "js": [
                "JS ke baare mein baat karte hain. Ye ek beautiful language hai, aur ye browser environment mein execute hoti hai. ",
                "Let's talk about JS today. Mai personally JS mein bahut saare projects karta hoon. ",
                "JS ek amazing language hai. Ismein aap browser mein bhi code likh sakte ho aur backend mein bhi. "
            ],
            "react": [
                "React mein aap declarative way mein UI design karte ho. Components ka concept hai jo reusable hota hai. ",
                "React actually Facebook ne develop kiya tha. Ye UI library hai jo Javascript ka power use karta hai. ",
                "React mein virtual DOM concept hai jo performance ko improve karta hai. Let me explain you in detail. "
            ],
            "node": [
                "Node.js actually Javascript ko browser se bahar leke aaya. Ye event-driven, non-blocking I/O model use karta hai. ",
                "Node js ek runtime environment hai, jo Chrome ke V8 engine pe built hai. Ismein aap server-side code likh sakte ho. ",
                "Node.js amazing hai backend ke liye. Ismein single thread pe multiple connections handle kar sakte ho. "
            ],
            "web": [
                "Web development mein aapko HTML, CSS aur Javascript - ye teen technologies aani chahiye. ",
                "Web development mein frontend aur backend dono important hai. Mai personally full stack development recommend karta hoon. ",
                "Web development ek beautiful journey hai. Ismein learning kabhi khatam nahi hoti. "
            ]
        }
        
        # Find which tech term was mentioned
        mentioned_tech = next((term for term in tech_terms.keys() if term in prompt.lower()), None)
        
        if mentioned_tech:
            intro = random.choice(tech_terms[mentioned_tech])
        else:
            # General tech intro with Piyush's style
            general_tech_intros = [
                "Programming mein concepts important hain. Mai hamesha kehta hoon fundamentals strong rakho, baaki sab aata rahega. ",
                "Software development mein problem solving skills matter karte hain. Mai recommend karta hoon DSA pe focus karo. ",
                "Tech field mein growth ke liye consistency zaroori hai. Mai bhi daily practice karta hoon aur learn karta hoon. "
            ]
            intro = random.choice(general_tech_intros)
    else:
        # Non-tech questions get Piyush's casual style
        casual_intros = [
            "Acha question hai! Chaliye discuss karte hain. ",
            "Interesting question hai ye! Mai aapko batata hoon. ",
            "Let's talk about this. Mai apna perspective share karta hoon. ",
            "Ye topic actually bahut interesting hai. Chaliye discuss karte hain. ",
            "Good question! Mai iske baare mein apne thoughts share karta hoon. "
        ]
        intro = random.choice(casual_intros)
    
    # Process content to match Piyush's speaking style
    transformed_sentences = []
    transformed_sentences.append(intro)
    
    # Pattern for inserting Piyush's phrases
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        # Skip very short sentences
        if len(sentence.split()) < 3:
            transformed_sentences.append(sentence)
            continue
        
        # Naturally insert Piyush's expressions based on sentence position
        if i == 0 or i % 4 == 0:  # Every few sentences add his flavor
            piyush_patterns = [
                # Emphasis patterns
                (r'\bimportant|crucial|key|essential', ["Actually ye bahut important hai. ", "Ye point note kar lijiye. ", "This is very very important. "]),
                # Example patterns
                (r'\bfor example|instance|such as', ["For example, ", "Matlab, ", "Jaise ki, "]),
                # Explanation patterns
                (r'\bmeans|meaning|implies', ["Matlab ", "Iska matlab ", "Simply speaking, "]),
                # Technical term patterns
                (r'\bfunction|method|algorithm|api', ["Actually ye ", "Basically ye ", "In technical terms, ye "]),
            ]
            
            modified = False
            for pattern, replacements in piyush_patterns:
                if re.search(pattern, sentence, re.IGNORECASE) and not modified:
                    if random.random() < 0.7:  # 70% chance
                        sentence = random.choice(replacements) + sentence
                        modified = True
            
            # If no patterns matched but we still want his flavor
            if not modified and random.random() < 0.3:
                piyush_fillers = [
                    "Let me tell you, ",
                    "Basically, ",
                    "Actually, ",
                    "Simply put, ",
                    "Let me explain, "
                ]
                sentence = random.choice(piyush_fillers) + sentence
        
        transformed_sentences.append(sentence)
    
    # Join the transformed sentences
    result = " ".join(transformed_sentences)
    
    # Add remaining paragraphs
    if len(paragraphs) > 1:
        additional_content = '\n\n'.join(paragraphs[1:])
        result += f"\n\n{additional_content}"
    
    # Add Piyush's signature conclusions
    if len(result) > 300 and random.random() < 0.8:
        conclusions = [
            "\n\nToh basically yahi hai iske baare mein. I hope aapko ye samajh mein aaya hoga.",
            "\n\nToh friends, ye tha aaj ka topic. Agar aapko kuch aur puchna hai toh feel free to ask.",
            "\n\nToh basically yahi hai main concept. Practice karte raho aur grow karte raho.",
            "\n\nSo I hope ye explanation clear tha. Koi doubts ho toh zaroor puchiye.",
            "\n\nSo friends, ye tha aaj ka concept. Keep learning, keep growing!"
        ]
        result += random.choice(conclusions)
    
    # Add motivation for programming questions
    if is_tech_question and random.random() < 0.6:
        tech_motivations = [
            "\n\nMai hamesha kehta hoon, projects banao. Real learning projects se hi hoti hai.",
            "\n\nMai recommend karta hoon documentation padho aur practice karo. Consistency is the key!",
            "\n\nTech field mein growth ke liye sirf watching tutorials enough nahi hai. Implement karo concepts ko.",
            "\n\nRemember, programming sikhne ke liye best way hai - solving real problems."
        ]
        result += random.choice(tech_motivations)
    
    # Occasionally add Piyush's signature phrases
    if random.random() < 0.3:
        signatures = [
            "\n\nHappy coding! 💻",
            "\n\nKeep learning, keep growing! 🚀",
            "\n\nBas practice karte raho, success milega! 💪",
            "\n\nConsistency is the key to success! 🔑"
        ]
        result += random.choice(signatures)
    
    return result

def transform_to_adaptive_style(prompt, content):
    """Transform the content to match the user's communication style with more natural variation"""
    metrics = get_user_metrics()
    
    # For very short prompts, just return the content directly
    if len(prompt.split()) <= 2:
        return content
    
    # Split content for processing
    paragraphs = content.split('\n\n')
    main_content = paragraphs[0] if paragraphs else content
    
    # Add greeting based on user's style
    greeting = ""
    if metrics["formal_language"] > 0.5:
        formal_greetings = [
            "I appreciate your question. ",
            "Thank you for your inquiry. ",
            "It's a pleasure to address this topic. ",
            ""  # Sometimes no greeting for variety
        ]
        greeting = random.choice(formal_greetings)
    else:
        casual_greetings = [
            "Thanks for asking! ",
            "Great question! ",
            "Happy to help with this! ",
            ""  # Sometimes no greeting for variety
        ]
        greeting = random.choice(casual_greetings)
    
    # Add personalization based on chat history (if it exists)
    personalization = ""
    if st.session_state.user_profile["message_count"] > 2:
        # Only add personalization sometimes
        if random.random() < 0.3:
            if metrics["top_topics"]:
                personalization = f"I see you've been interested in {metrics['top_topics'][0]}. "
    
    # Combine greeting, personalization, and content
    result = f"{greeting}{personalization}{main_content}"
    
    # Add remaining paragraphs
    if len(paragraphs) > 1:
        additional_content = '\n\n'.join(paragraphs[1:])
        result += f"\n\n{additional_content}"
    
    # Add emoji if user uses them (but not every time)
    if metrics["emoji_usage"] > 0.3 and random.random() < 0.4:
        emojis = ["👍", "✨", "💡", "🤔", "👋", "🚀"]
        result += f" {random.choice(emojis)}"
    
    # Add a question if the user asks questions frequently (but not every time)
    if metrics["question_ratio"] > 0.5 and random.random() < 0.3:
        questions = [
            "\n\nDoes that answer your question?",
            "\n\nWhat do you think about this approach?",
            "\n\nWould you like me to elaborate on any specific part?"
        ]
        result += random.choice(questions)
    
    return result

def transform_to_formal_style(prompt, content):
    """Transform the content to a formal, professional style with natural variation"""
    # For very short queries, return content as is
    if len(prompt.split()) <= 3:
        return content
    
    # Split content into paragraphs for processing
    paragraphs = content.split('\n\n')
    main_content = paragraphs[0] if paragraphs else content
    
    # Vary formal introductions based on question type
    is_technical = any(word in prompt.lower() for word in ["javascript", "python", "code", "programming", "tech", "framework"])
    is_inquiry = any(word in prompt.lower() for word in ["what", "how", "why", "when", "where", "explain"])
    
    # Select appropriate formal introduction
    if is_technical and is_inquiry:
        formal_intros = [
            "Regarding your technical inquiry, ",
            "In response to your question about this technical matter, ",
            "With respect to your technical question, ",
            ""  # Sometimes no intro for variety
        ]
    elif is_technical:
        formal_intros = [
            "On the subject of this technical matter, ",
            "With regard to this technical topic, ",
            "Concerning this technical subject, ",
            ""  # Sometimes no intro for variety
        ]
    elif is_inquiry:
        formal_intros = [
            "In response to your inquiry, ",
            "Regarding your question, ",
            "With respect to your query, ",
            ""  # Sometimes no intro for variety
        ]
    else:
        formal_intros = [
            "Thank you for your message. ",
            "I appreciate your interest in this matter. ",
            "With regard to the topic at hand, ",
            ""  # Sometimes no intro for variety
        ]
    
    intro = random.choice(formal_intros)
    
    # For longer content, apply formal transformation with natural variations
    if len(main_content) > 200:
        # Break the content into sentences
        sentences = re.split(r'(?<=[.!?])\s+', main_content)
        
        # Transform some sentences to be more formal
        formal_sentences = []
        formal_sentences.append(intro)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Expand contractions (only some of the time)
            if random.random() < 0.6:
                contractions = {
                    "isn't": "is not",
                    "aren't": "are not",
                    "wasn't": "was not",
                    "weren't": "were not",
                    "don't": "do not",
                    "doesn't": "does not",
                    "didn't": "did not",
                    "can't": "cannot",
                    "couldn't": "could not",
                    "shouldn't": "should not",
                    "wouldn't": "would not",
                    "won't": "will not",
                    "I'm": "I am",
                    "you're": "you are",
                    "he's": "he is",
                    "she's": "she is",
                    "it's": "it is",
                    "we're": "we are",
                    "they're": "they are"
                }
                
                for contraction, expanded in contractions.items():
                    sentence = re.sub(r'\b' + contraction + r'\b', expanded, sentence, flags=re.IGNORECASE)
            
            # Apply some formal word replacements (only occasionally)
            if random.random() < 0.3:
                replacements = {
                    " use": " utilize",
                    " make": " construct",
                    " think": " consider",
                    " look at": " examine",
                    " help": " assist",
                    " start": " initiate",
                    " end": " conclude"
                }
                
                # Apply at most one replacement per sentence for natural variation
                applied = False
                for casual, formal in replacements.items():
                    if casual in sentence.lower() and not applied:
                        pattern = r'\b' + casual.strip() + r'\b'
                        sentence = re.sub(pattern, formal.strip(), sentence, flags=re.IGNORECASE, count=1)
                        applied = True
            
            formal_sentences.append(sentence)
        
        # Join the transformed sentences
        result = " ".join(formal_sentences)
    else:
        # For shorter content, just add the formal intro
        result = f"{intro}{main_content}"

    if re.search(r'\bwho are you\b|\bwhat is your name\b|\bwho am i talking to\b', prompt.lower()):
        return "I am a professional educator specializing in technology and programming instruction. My goal is to provide clear, thorough, and structured learning content to help you develop your technical skills. How may I assist with your educational needs today?"
    
    # Add remaining paragraphs
    if len(paragraphs) > 1:
        additional_content = '\n\n'.join(paragraphs[1:])
        result += f"\n\n{additional_content}"
    
    # Sometimes add a formal closing (but not always)
    if random.random() < 0.4:
        formal_closings = [
            "\n\nPlease do not hesitate to inquire should you require any further clarification.",
            "\n\nI trust this information proves satisfactory for your needs.",
            "\n\nShould you have any additional questions, I remain at your disposal."
        ]
        result += random.choice(formal_closings)
    
    return result

if __name__ == "__main__":
    main()