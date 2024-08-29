import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


def get_openai_response(prompt, api_key):
    """Generate a response from OpenAI's GPT model based on the provided prompt."""
    openai.api_key = api_key
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # You can change this to a different model if needed
            messages=[
    {"role": "system", "content": "You are consultent."},
    {"role": "user", "content": prompt}
  ],
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {e}"


def main():
    st.title("Car Price Prediction App")

    # Sidebar for OpenAI API Key
    st.sidebar.header("OpenAI API Key")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    st.write("Please enter the details of the car to get the price prediction.")

    # Collect input from the user
    model = st.selectbox("Model Name",['Alto', 'Grand', 'i20', 'Ecosport', 'Wagon R', 'i10', 'Venue',
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

    # Button to make prediction
    if st.button("Predict"):
        # Create an instance of CustomData with the user input
        custom_data = CustomData(
            model = model,
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
        st.markdown(
    f"<h3 style='color: #FF6347;'>Predicted Price: <span style='color: #32CD32;'>₹{prediction[0]:,.2f}</span></h3>",
    unsafe_allow_html=True
)

        # Generate and display response from OpenAI if API key is provided
        if api_key:
            prompt = ( f"""
You are a seasoned automotive consultant with extensive experience in evaluating used car prices. Based on the prediction for a {model} with the following details:

- **Model:** {model}
- **Vehicle Age:** {vehicle_age} years
- **KM Driven:** {km_driven} km
- **Mileage:** {mileage} km/l
- **Engine:** {engine} cc
- **Max Power:** {max_power} bhp
- **Seats:** {seats}
- **Seller Type:** {seller_type}
- **Fuel Type:** {fuel_type}
- **Transmission Type:** {transmission_type}

The predicted price is ₹{prediction[0]:,.2f}.

Please provide a detailed analysis including:

1. **Market Position:** Compare this predicted price with similar vehicles currently available in the market. Are there any specific trends, brand value, or market conditions affecting this price?

2. **Price Justification:** Explain the factors that contribute to this predicted price. Consider the car’s age, condition, mileage, and features. How do these aspects affect the value of the vehicle in the current market?

3. **Actionable Insights:** Provide practical advice on what actions the user can take to either increase the car’s value or negotiate effectively if they are buying or selling this vehicle.

The goal is to help the user make well-informed decisions regarding their car based on the prediction and current market conditions.
"""
)

            response = get_openai_response(prompt, api_key)
            st.write("Response from OpenAI:")
            st.write(response)
        else:
            st.write("Please provide your OpenAI API key to get a detailed response.")


if __name__ == "__main__":
    main()
