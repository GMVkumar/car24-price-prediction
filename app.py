import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# Load Trained XGBoost Model
# =====================================
model = joblib.load("xgboost_model.pkl")

# =====================================
# Streamlit App Title & Description
# =====================================
st.title("🚗 Car Price Prediction App")
st.write("Predict the selling price of a car based on its details using an XGBoost model.")

# =====================================
# Brand → Model Mapping
# =====================================
brand_models = {
    'Maruti': ['Alto', 'Wagon R', 'Swift', 'Swift Dzire', 'Ciaz', 'Baleno', 'Vitara', 'Celerio', 'S-Presso', 'Eeco', 'XL6', 'Glanza'],
    'Hyundai': ['Grand', 'i20', 'i10', 'Verna', 'Venue', 'Creta', 'Elantra', 'Aura', 'Santro'],
    'Ford': ['Ecosport', 'Aspire', 'Figo', 'Freestyle', 'Endeavour'],
    'Renault': ['Duster', 'KWID', 'Triber'],
    'Mercedes-Benz': ['C-Class', 'E-Class', 'S-Class', 'CLS'],
    'Toyota': ['Innova', 'Fortuner', 'Camry'],
    'Volkswagen': ['Vento', 'Polo'],
    'Honda': ['City', 'Amaze', 'Jazz', 'CR-V', 'Civic', 'WR-V'],
    'Mahindra': ['Bolero', 'Scorpio', 'XUV500', 'KUV100', 'KUV', 'Marazzo', 'Thar'],
    'Datsun': ['GO', 'RediGO', 'redi-GO'],
    'Tata': ['Tiago', 'Tigor', 'Safari', 'Nexon', 'Harrier', 'Altroz', 'Hexa'],
    'Kia': ['Seltos'],
    'Audi': ['A4', 'A6', 'Q7'],
    'Mini': ['Cooper'],
    'Isuzu': ['D-Max'],
    'BMW': ['3', '5', '7', 'X1', 'X3', 'X5'],
    'Skoda': ['Rapid', 'Superb', 'Octavia'],
    'Volvo': ['XC', 'XC60'],
    'Nissan': ['Kicks', 'X-Trail'],
    'Land Rover': ['Rover'],
    'Jeep': ['Compass'],
    'Jaguar': ['XF'],
    'Force': ['Gurkha']
}

# =====================================
# Input Layout
# =====================================
col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Select Brand", list(brand_models.keys()))
    model_name = st.selectbox("Select Model", brand_models[brand])
    vehicle_age = st.slider("Car Age (in years)", 0, 20, 5)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000)

with col2:
    seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    transmission_type = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
    mileage = st.number_input("Mileage (km/l or km/kg)", min_value=0.0, max_value=50.0, value=18.0)

with col3:
    engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, step=100)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=600.0, value=80.0)
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8])

# =====================================
# Encoding (Match Training Encodings)
# =====================================
brand_mapping = {b: i for i, b in enumerate(brand_models.keys())}
fuel_mapping = {f: i for i, f in enumerate(['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])}
transmission_mapping = {'Manual': 0, 'Automatic': 1}
seller_mapping = {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}

# Simple encoding for model names (stable hash)
model_encoded = abs(hash(model_name)) % 1000

# =====================================
# Prepare Input DataFrame
# =====================================
input_data = pd.DataFrame({
    'brand': [brand_mapping.get(brand, 0)],
    'model': [model_encoded],
    'vehicle_age': [vehicle_age],
    'km_driven': [km_driven],
    'seller_type': [seller_mapping.get(seller_type, 0)],
    'fuel_type': [fuel_mapping.get(fuel_type, 0)],
    'transmission_type': [transmission_mapping.get(transmission_type, 0)],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats]
})

# =====================================
# Prediction Section
# =====================================
st.markdown("---")
if st.button("🔍 Predict Price"):
    try:
        pred_log_price = model.predict(input_data)[0]
        pred_price = np.expm1(pred_log_price)  # reverse log1p transform
        st.success(f"💰 **Estimated Car Price:** ₹{pred_price:,.0f}")
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")

# =====================================
# Footer
# =====================================
st.caption("Model trained using XGBoost with hyperparameter tuning on the CarDekho dataset.")
