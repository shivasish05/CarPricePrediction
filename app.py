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
    {"role": "system", "content": "You are a helpful assistant, based of predicted price give suggestions of ."},
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
        st.markdown(f"Predicted Price: {prediction[0]:,.2f}")

        # Generate and display response from OpenAI if API key is provided
        if api_key:
            prompt = (f"""The predicted price for a car with the following details is ${prediction[0]:,.2f}:

- **Vehicle Age:** {vehicle_age} years
- **KM Driven:** {km_driven} km
- **Mileage:** {mileage} km/l
- **Engine:** {engine} cc
- **Max Power:** {max_power} bhp
- **Seats:** {seats}
- **Seller Type:** {seller_type}
- **Fuel Type:** {fuel_type}
- **Transmission Type:** {transmission_type}

Please provide a detailed response addressing the following:

1. **Market Comparison:** How does this predicted price compare to similar cars currently on the market? Are there any trends or patterns that influence this price?
2. **Value Proposition:** What factors contribute to this predicted price? How do the car’s features and condition impact the price?
3. **Actionable Advice:** For a buyer, what should they consider based on this prediction? For a seller, what strategies should they use to maximize their selling price?
4. **Improvement Tips:** Are there any potential improvements or modifications that could increase the car’s value?
5. **Risk Factors:** What are the potential risks or uncertainties associated with this prediction?

Provide a comprehensive and actionable explanation to help users understand and make informed decisions based on this prediction.
""")

            response = get_openai_response(prompt, api_key)
            st.write("Response from OpenAI:")
            st.write(response)
        else:
            st.write("Please provide your OpenAI API key to get a detailed response.")


if __name__ == "__main__":
    main()
