# Importing dependencies:

import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor

# Create a Streamlit app
st.write("""
# MSDE5 : End of Studies Project

## Prediction of the PH of treated Water in the Jorf Lasfar Desalination Plant

""")

st.sidebar.image(r"C:\Users\user\Desktop\PFE\desalination.jpg", width=300)
st.sidebar.header('User Input Parameters')

# Define input features
def user_input_features():
    Conductivity = st.sidebar.slider("Conductivity", 0, 100)
    PH_of_the_solution = st.sidebar.slider("PH of the solution", 0, 14)
    PH_of_water_backwash = st.sidebar.slider("PH of water backwash", 0, 14)
    PH_of_seawater = st.sidebar.slider("PH of seawater", 0, 14)
    Pressure_of_water_entering_membrane_1 = st.sidebar.slider("Pressure of water entering membrane 1", 0, 100)
    Pressure_of_water_entering_membrane_2 = st.sidebar.slider("Pressure of water entering membrane 2", 0, 100)
    Pressure_of_water_entering_membrane_3 = st.sidebar.slider("Pressure of water entering membrane 3", 0, 100)
    Pressure_of_water_entering_membrane_4 = st.sidebar.slider("Pressure of water entering membrane 4", 0, 100)
    Pressure_of_water_entering_membrane_5 = st.sidebar.slider("Pressure of water entering membrane 5", 0, 100)
    Turbidity = st.sidebar.slider("Turbidity", 0, 12)
    Temperature = st.sidebar.slider("Temperature", 0, 40)
    
    data = {'Feature': ['Conductivity', 'PH of the solution', 'PH of water backwash', 'PH of seawater',
                        'Pressure of water entering membrane 1', 'Pressure of water entering membrane 2',
                        'Pressure of water entering membrane 3', 'Pressure of water entering membrane 4',
                        'Pressure of water entering membrane 5', 'Turbidity', 'Temperature'],
            'Value': [Conductivity, PH_of_the_solution, PH_of_water_backwash, PH_of_seawater,
                      Pressure_of_water_entering_membrane_1, Pressure_of_water_entering_membrane_2,
                      Pressure_of_water_entering_membrane_3, Pressure_of_water_entering_membrane_4,
                      Pressure_of_water_entering_membrane_5, Turbidity, Temperature]}
    
    features = pd.DataFrame(data)
    return features

# Load your pre-trained model (RandomForestRegressor in this case)
model_path = r"C:\Users\user\Desktop\PFE\best_model.pkl"
model = joblib.load(model_path)

# Main app logic
if __name__ == '__main__':
    df = user_input_features()

    st.subheader('User Input parameters')
    st.table(df.set_index('Feature'))  # Display input features in a table with headers 'Feature' and 'Value'
    
    # Prepare input data for prediction
    input_data = df['Value'].values.reshape(1, -1)  # Reshape input data for prediction
    
    # Make predictions
    if st.sidebar.button('Submit'):
        prediction = model.predict(input_data)
        st.write(f'Predicted PH of Treated Water: {prediction[0]}')



st.write("""
---
**Khawla BADDAR, MSc**  
Master's in Data Engineering  
""")