from fastapi import FastAPI
from pydantic import BaseModel
from backend.ev_guardian_brain import EVGuardianBrain
from tensorflow.keras.models import load_model
import joblib

model = load_model("backend/models/ev_soc_lstm_model.keras", compile=False)
scaler= joblib.load("backend/models/ev_feature_scaler.pkl")

brain = EVGuardianBrain(
    model=model,
    scaler=scaler,
    charging_stations_km=[20,55,90],
    safety_margin_km=10
)

class StepInput(BaseModel):
    speed_kmph:float
    current_soc:float
    battery_current:float
    battery_voltage:float
    battery_temp:float
    dt_sec:float = 1.0



app =  FastAPI()
@app.post("/step")
def step_ev(data: StepInput):
    status =brain.step(
        speed_kmph=data.speed_kmph,
        dt_sec=data.dt_sec,
        current_soc=data.current_soc,
        battery_current=data.battery_current,
        battery_voltage=data.battery_voltage,
        battery_temp=data.battery_temp

    )
    return status
@app.get("/")
def root():
    return {
        "status": "running",
        "service": "EV Battery Guardian API"
    }
