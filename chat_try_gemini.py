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

# Load bot and user images
bot_image = "bot_image.png"  # Replace with your bot image path
user_image = "/img/hitesh.jpg" # Replace with your user image path

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
    # Set the page title and configure the layout
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
    
    # Track when greetings were last used to avoid repetition
    if "last_greeting_index" not in st.session_state:
        st.session_state.last_greeting_index = -1
        
    if "style_elements_used" not in st.session_state:
        st.session_state.style_elements_used = set()
    
    # Set up Google Gemini API from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
        st.stop()
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Set up the sidebar configuration
    with st.sidebar:
        st.title("Google Gemini Configuration")
        st.success("✅ API Key loaded from environment variable")
        
        st.divider()
        
        # Model selection
        model = st.selectbox(
            "Select Gemini Model",
            [ "gemini-1.5-pro", "gemini-1.5-flash"]
        )
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values make it more deterministic"
        )
        
        st.divider()
        
        # User profile metrics display (if available)
        if st.session_state.user_profile["message_count"] > 0:
            st.subheader("User Profile Metrics")
            metrics = get_user_metrics()
            st.write(f"Avg Sentiment: {metrics['avg_sentiment']:.2f}")
            st.write(f"Technical Language: {metrics['technical_ratio']:.2f}")
            st.write(f"Formality Level: {metrics['formal_language']:.2f}")
            st.write(f"Question Ratio: {metrics['question_ratio']:.2f}")
            st.write(f"Uses Emojis: {'Yes' if metrics['emoji_usage'] > 0.3 else 'No'}")
            if metrics['top_topics']:
                st.write(f"Top Topics: {', '.join(metrics['top_topics'][:3])}")

    st.title("Adaptive Chat with Chai with Code Mentors")

    # Only show the radio button if no option has been selected yet
    if not st.session_state.option_selected:
        st.write("Welcome! Please select one of the three options below:")
        
        option = st.radio(
            "Choose an option:",
            ["Option 1: Hitesh Style", "Option 2: Adaptive Style", "Option 3: Formal Style"],
            index=None
        )
        
        if option:
            if st.button(f"Confirm {option}"):
                st.session_state.option_selected = option
                st.session_state.messages = []  # Clear any previous chat
                # Reset style tracking
                st.session_state.last_greeting_index = -1
                st.session_state.style_elements_used = set()
                st.rerun()  # Rerun the app to update the interface
    
    # If an option is selected, show the chat interface
    else:
        option = st.session_state.option_selected
        
        # Show which option was selected with a way to reset
        col1, col2 = st.columns([3, 1])
        with col1:
            if "Option 1" in option:
                st.success(f"You selected: {option}")
                st.caption("Responses will use content from Google Gemini in Hitesh's distinctive style")
            elif "Option 2" in option:
                st.info(f"You selected: {option}")
                st.caption("Responses will use content from Google Gemini adapted to match your communication style")
            else:
                st.warning(f"You selected: {option}")
                st.caption("Responses will use content from Google Gemini in a formal, professional style")
        
        with col2:
            if st.button("Reset Selection"):
                st.session_state.option_selected = None
                st.session_state.messages = []
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
                # Reset style tracking
                st.session_state.last_greeting_index = -1
                st.session_state.style_elements_used = set()
                st.rerun()
        
        # Display chat messages from history
        st.divider()
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Accept user input
        if prompt := st.chat_input("What would you like to talk about?"):
            # Analyze user input and update profile
            analyze_user_input(prompt)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get raw content from Google Gemini
            try:
                with st.spinner("Getting response from Google Gemini..."):
                    raw_content = get_gemini_response(prompt, model, temperature)
                    
                # Transform the content based on selected option
                with st.chat_message("assistant"):
                    if "Option 1" in option:
                        response = transform_to_hitesh_style(prompt, raw_content)
                    elif "Option 2" in option:
                        response = transform_to_adaptive_style(prompt, raw_content)
                    else:
                        response = transform_to_formal_style(prompt, raw_content)
                    
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            except Exception as e:
                st.error(f"Error communicating with Google Gemini: {str(e)}")

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
    is_greeting = any(word in prompt.lower() for word in ["hello", "hi", "hey", "namaste"])
    is_tech_question = any(word in prompt.lower() for word in ["javascript", "js", "react", "code", "coding", "web", "html", "css", "programming"])
    is_short_query = len(prompt.split()) <= SHORT_QUERY_THRESHOLD

    # Context-aware introduction selection
    hitesh_intros = [
        "Namaste friends! ",
        "Hello everyone! ",
        "What's up guys! ",
        "Hey there! ",
        "Aaj hum baat karenge ",
        "Chalo shuru karte hain ",
        ""  # Empty intro for some responses to sound more natural
    ]
    
    # Track which Hindi expressions have been used recently
    hindi_expressions = [
        "Bilkul",
        "Ekdum sahi",
        "Bahut badhiya",
        "Aap samajh rahe ho?",
        "Dekho",
        "Matlab",
        "Aur ye important hai",
        "Ye samajh lo",
        "Dhyaan se suno"
    ]
    
    # Choose introduction based on context
    if is_greeting:
        # For greetings, pick a greeting response that hasn't been used recently
        greeting_responses = [
            "Namaste! Kaise ho aap? ",
            "Hanji! Kaise ho aap! ",
            "Hey! Great to see you here! ",
            "Hi there! Chai aur Code me aapka swagat hai! "
        ]
        
        # Avoid using the same greeting twice in a row
        available_indexes = list(range(len(greeting_responses)))
        if st.session_state.last_greeting_index in available_indexes:
            available_indexes.remove(st.session_state.last_greeting_index)
        
        greeting_index = random.choice(available_indexes)
        st.session_state.last_greeting_index = greeting_index
        
        return greeting_responses[greeting_index] + "How can I help you today?"
    
    # For short queries, don't add style elements unless it's a tech question
    if is_short_query and not is_tech_question:
        return content
    
    # Get the first paragraph of content for the core information
    paragraphs = content.split('\n\n')
    main_content = paragraphs[0] if paragraphs else content
    
    # For technical questions, use Hitesh's teaching style
    if is_tech_question:
        # Check what tech term was mentioned
        tech_terms = ["javascript", "js", "react", "code", "coding", "web", "html", "css", "programming"]
        mentioned_tech = next((term for term in tech_terms if term in prompt.lower()), None)
        
        # Create a context-specific introduction
        if mentioned_tech:
            tech_specific_intros = {
                "javascript": [
                    f"JavaScript ek powerful language hai! ",
                    f"Aaj hum JavaScript ke baare mein baat karenge. ",
                    f"JavaScript fundamentals samajhna bahut zaroori hai. "
                ],
                "js": [
                    f"JS ek versatile language hai! ",
                    f"JS ko master karna chahte ho? Great! ",
                    f"Aaj hum JS ke baare mein baat karenge. "
                ],
                "react": [
                    f"React ek revolutionary frontend library hai! ",
                    f"React ke saath development bahut maza aata hai. ",
                    f"Aaj hum React ke baare mein baat karenge. "
                ],
                "web": [
                    f"Web development mein sabse important hai fundamentals. ",
                    f"Web ke liye aapko HTML, CSS aur JS teeno aani chahiye. ",
                    f"Web development is all about practice! "
                ],
                "html": [
                    f"HTML is the skeleton of every website! ",
                    f"HTML ke bina web development possible hi nahi hai. ",
                    f"Aaj hum HTML ke baare mein baat karenge. "
                ],
                "css": [
                    f"CSS is what makes websites beautiful! ",
                    f"CSS mein practice hi sabse zaroori hai. ",
                    f"CSS ke baare mein aaj baat karenge. "
                ]
            }
            
            # Use specific intro or fallback to general coding intro
            specific_intros = tech_specific_intros.get(mentioned_tech, [
                "Coding ke liye mindset bahut zaroori hai. ",
                "Programming sikhne ke liye consistency chahiye. ",
                "As a developer, you need to understand core concepts first. "
            ])
            
            intro = random.choice(specific_intros)
        else:
            # General tech intro
            intro = random.choice([
                "Tech world mein aage badhne ke liye concepts clear hone chahiye. ",
                "Programming mein practice hi key hai. ",
                "Development skills improve karne ke liye projects banao. "
            ])
    else:
        # Non-tech questions get a more generic intro
        intro = random.choice(hitesh_intros)
    
    # Special styling for longer content
    if len(main_content) > 300:
        # Break up long content into segments
        segments = re.split(r'(?<=[.!?])\s+', main_content)
        
        # Enhanced segments with Hitesh's style
        enhanced_segments = []
        
        # Add intro
        enhanced_segments.append(intro)
        
        # Only apply style elements to some segments (not all, to keep it natural)
        for i, segment in enumerate(segments):
            if not segment.strip():
                continue
                
            # Skip very short segments
            if len(segment.split()) < 3:
                enhanced_segments.append(segment)
                continue
            
            # Only apply style to selected segments (about 30%)
            if random.random() < 0.3:
                style_applied = False
                
                # Add Hindi expression (but not too frequently)
                if random.random() < 0.2:
                    # Choose an expression that hasn't been used recently
                    available_expressions = [e for e in hindi_expressions if e not in st.session_state.style_elements_used]
                    if not available_expressions:  # If all have been used, reset
                        available_expressions = hindi_expressions
                        st.session_state.style_elements_used = set()
                    
                    expression = random.choice(available_expressions)
                    st.session_state.style_elements_used.add(expression)
                    
                    enhanced_segments.append(f"{expression}! {segment}")
                    style_applied = True
                
                # Add tech emphasis for some segments
                elif random.random() < 0.15 and is_tech_question:
                    tech_emphasis = [
                        "Ye concept industry mein bahut use hota hai. ",
                        "Real projects mein aise hi kaam hota hai. ",
                        "Interview mein ye zaroor puchha jata hai. "
                    ]
                    enhanced_segments.append(f"{segment} {random.choice(tech_emphasis)}")
                    style_applied = True
                    
                if not style_applied:
                    enhanced_segments.append(segment)
            else:
                enhanced_segments.append(segment)
        
        # Join the enhanced segments
        result = " ".join(enhanced_segments)
    else:
        # For shorter content, apply light styling
        result = f"{intro}{main_content}"
    
    # Add remaining paragraphs
    if len(paragraphs) > 1:
        additional_content = '\n\n'.join(paragraphs[1:])
        result += f"\n\n{additional_content}"
    
    # Sometimes add a Hitesh-style conclusion
    if random.random() < 0.3 and not is_short_query:
        conclusions = [
            "\n\nAur yaad rakhiye, practice is the key to success! Keep coding!",
            "\n\nSo that's it for today! Aur koi sawal ho to comment section mein puchh sakte ho!",
            "\n\nI hope yeh explanation clear hui! Agar aapko ye video helpful lagi to like zaroor karna!",
            "\n\nAb aapko samajh mein aa gaya hoga. Keep coding, keep exploring, aur chai peete raho! ☕"
        ]
        result += random.choice(conclusions)
    
    # Add occasional emoji for emphasis, but not too many
    if random.random() < 0.2 and "☕" not in result and "🔥" not in result:
        emojis = ["🔥", "💻", "👨‍💻", "✅", "👍", "💡"]
        result = result.rstrip() + f" {random.choice(emojis)}"
    
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