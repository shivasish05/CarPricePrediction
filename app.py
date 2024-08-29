import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

def get_openai_response(messages, api_key):
    """Generate a response from OpenAI's GPT model based on the conversation history."""
    openai.api_key = api_key
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Ensure the correct model name
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def main():
    st.set_page_config(page_title="Car Price Prediction App", page_icon="🚗", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #2C3E50;
            }
            .stApp {
                background-color: #2C3E50;
                color: white;
            }
            .sidebar .sidebar-content {
                background-color: #2C3E50;
                padding: 20px;
                border-radius: 10px;
            }
            .sidebar .sidebar-content h2 {
                color: #ECF0F1;
            }
            .sidebar .sidebar-content input {
                background-color: #34495E;
                color: white;
                border-radius: 5px;
                border: 1px solid #7F8C8D;
            }
            .sidebar .sidebar-content label {
                color: #ECF0F1;
            }
            .stButton button {
                background-color: #E74C3C;
                color: white;
                font-size: 16px;
                border-radius: 10px;
                padding: 10px 20px;
            }
            .stButton button:hover {
                background-color: #C0392B;
                color: white;
            }
            .chat-box {
                background-color: #34495E;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }
            .user-message {
                background-color: #1ABC9C;
                color: white;
                padding: 10px 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                max-width: 80%;
            }
            .assistant-message {
                background-color: #3498DB;
                color: white;
                padding: 10px 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                max-width: 80%;
            }
            .prediction-result {
                background-color: #27AE60;
                border-left: 6px solid #2ECC71;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                color: white;
            }
            .stTextInput > div > input {
                background-color: #34495E;
                color: white;
                border-radius: 10px;
                padding: 10px;
                border: 1px solid #7F8C8D;
            }
            .stTextInput > div > label {
                color: white;
            }
            h1, h2, h3, h4, h5, h6 {
                color: white;
            }
            label {
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚗 Car Price Prediction App")

    # Sidebar for OpenAI API Key
    st.sidebar.header("🔑 OpenAI API Key")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    st.write("### Please enter the details of the car to get the price prediction.")

    # Collect input from the user
    model = st.selectbox("Model Name", ['Alto', 'Grand', 'i20', 'Ecosport', 'Wagon R', 'i10', 'Venue',
       'Swift', 'Verna', 'Duster', 'Cooper', 'Ciaz', 'C-Class', 'Innova',
       'Baleno', 'Swift Dzire', 'Vento', 'Creta', 'City', 'Bolero',
       'Fortuner', 'KWID', 'Amaze', 'Santro', 'XUV500', 'KUV100', 'Ignis',
       'RediGO', 'Scorpio', 'Marazzo', 'Aspire', 'Figo', 'Vitara',
       'Tiago', 'Polo', 'Seltos', 'Celerio', 'GO', '5', 'CR-V',
       'Endeavour', 'KUV', 'Jazz', '3', 'A4', 'Tigor', 'Ertiga', 'Safari',
       'Thar', 'Hexa', 'Rover', 'Eeco', 'A6', 'E-Class', 'Q7', 'Z4', '6',
       'XF', 'X5', 'Hector', 'Civic', 'D-Max', 'Cayenne', 'X1', 'Rapid',
       'Freestyle', 'Superb', 'Nexon', 'XUV300', 'Dzire VXI', 'S90',
       'WR-V', 'XL6', 'Triber', 'ES', 'Wrangler', 'Camry', 'Elantra',
       'Yaris', 'GL-Class', '7', 'S-Presso', 'Dzire LXI', 'Aura', 'XC',
       'Ghibli', 'Continental', 'CR', 'Kicks', 'S-Class', 'Tucson',
       'Harrier', 'X3', 'Octavia', 'Compass', 'CLS', 'redi-GO', 'Glanza',
       'Macan', 'X4', 'Dzire ZXI', 'XC90', 'F-PACE', 'A8', 'MUX',
       'GTC4Lusso', 'GLS', 'X-Trail', 'XE', 'XC60', 'Panamera', 'Alturas',
       'Altroz', 'NX', 'Carnival', 'C', 'RX', 'Ghost', 'Quattroporte',
       'Gurkha'])
    vehicle_age = st.text_input("Vehicle Age (e.g., 2 years)")
    km_driven = st.number_input("KM Driven", min_value=0.0, step=100.0)
    mileage = st.number_input("Mileage (in km/l)", min_value=0.0, step=0.1)
    engine = st.number_input("Engine (in cc)", min_value=0.0, step=100.0)
    max_power = st.number_input("Max Power (in bhp)", min_value=0.0, step=5.0)
    seats = st.number_input("Seats", min_value=2, step=1)
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "predicted" not in st.session_state:
        st.session_state.predicted = False

    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    # Button to make prediction
    if st.button("Predict"):
        # Create an instance of CustomData with the user input
        custom_data = CustomData(
            model=model,
            vehicle_age=vehicle_age,
            km_driven=km_driven,
            mileage=mileage,
            engine=engine,
            max_power=max_power,
            seats=seats,
            seller_type=seller_type,
            fuel_type=fuel_type,
            transmission_type=transmission_type
        )

        # Convert input data to DataFrame
        data_df = custom_data.get_data_as_dataframe()

        # Initialize prediction pipeline and get the prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data_df)

        # Show prediction result
        st.markdown(f"""
            <div class="prediction-result">
                <h4>Predicted Price: ₹{prediction[0]:,.2f}</h4>
            </div>
        """, unsafe_allow_html=True)

        st.session_state.predicted = True
        st.session_state.prediction = prediction[0]

        # Initialize chat history with prediction context
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful car consultent."},
            {"role": "user", "content": (
                f"The predicted price for a {model} with the following details is ₹{prediction[0]:,.2f}:\n\n"
                f"- **Model:** {model}\n"
                f"- **Vehicle Age:** {vehicle_age} years\n"
                f"- **KM Driven:** {km_driven} km\n"
                f"- **Mileage:** {mileage} km/l\n"
                f"- **Engine:** {engine} cc\n"
                f"- **Max Power:** {max_power} bhp\n"
                f"- **Seats:** {seats}\n"
                f"- **Seller Type:** {seller_type}\n"
                f"- **Fuel Type:** {fuel_type}\n"
                f"- **Transmission Type:** {transmission_type}\n"
                
            )}
        ]

    # Real-time Chatbot interface (only available after prediction)
    if st.session_state.predicted:
        st.write("## Ask further questions about the prediction")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="user-message">**User:** {msg["content"]}</div>', unsafe_allow_html=True)
            elif msg['role'] == 'assistant':
                st.markdown(f'<div class="assistant-message">**Assistant:** {msg["content"]}</div>', unsafe_allow_html=True)

        # User input for chat
        user_message = st.text_input("Type your message:", key="chat_input_key", placeholder="Ask something about the prediction...")

        if st.button("Send"):
            if user_message:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                
                if api_key:
                    with st.spinner('Thinking...'):
                        # Get OpenAI response
                        assistant_response = get_openai_response(st.session_state.chat_history, api_key)
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                        st.markdown(f'<div class="assistant-message">**Assistant:** {assistant_response}</div>', unsafe_allow_html=True)
                else:
                    st.write("Please provide your OpenAI API key to get responses.")
            
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.write("Chat history cleared.")

if __name__ == "__main__":
    main()
