from typing import List
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
    safety_margin_km=10
)


class EVinput(BaseModel):
    speed:float
    battery_percentage:float
    distance_travelled:float
    temperature:float
class station(BaseModel):
    distance_km:float
    name:str

class AnalysisResponse(BaseModel):
    predicted_range_km:float
    warning:bool
    warning_reason:str
    recommended_stations:List[station]

class StepInput(BaseModel):
    speed_kmph:float
    current_soc:float
    battery_current:float
    battery_voltage:float
    battery_temp:float
    dt_sec:float = 1.0

    current_lat: float | None = None
    current_lon: float | None = None




app =  FastAPI()
@app.post("/step")
def step_ev(data: StepInput):
    status =brain.step(
        speed_kmph=data.speed_kmph,
        dt_sec=data.dt_sec,
        current_soc=data.current_soc,
        battery_current=data.battery_current,
        battery_voltage=data.battery_voltage,
        battery_temp=data.battery_temp,
        current_lat=data.current_lat,
        current_lon=data.current_lon

    )
    return status

"""@app.post("/warn")
def warning_logic(
    predicted_range_km: float,
    nearest_station_km: float,
    safety_margin_km: float = 10
):
    if predicted_range_km < nearest_station_km + safety_margin_km:
        return {
            "warning": True,
            "reason": "Insufficient range to safely reach nearest charging station"
        }

    return {
        "warning": False,
        "reason": None
    }

@app.post("/recommend-stations")
def recommend_stations(
    predicted_range_km: float,
    stations: List[station],
    safety_margin_km: float = 10
):
    usable_range = predicted_range_km - safety_margin_km

    reachable = [
        s for s in stations if s.distance_km <= usable_range
    ]

    reachable.sort(key=lambda x: x.distance_km)

    return {
        "reachable_count": len(reachable),
        "stations": reachable
    }
@app.post("/full-analysis", response_model=AnalysisResponse)
def full_analysis(
    predicted_range_km: float,
    predicted_battery: float,
    stations: List[station],
    safety_margin_km: float = 10,
    critical_battery: float = 20
):
    usable_range = predicted_range_km - safety_margin_km

    reachable = [s for s in stations if s.distance_km <= usable_range]
    reachable.sort(key=lambda x: x.distance_km)

    warning = False
    reason = None

    if predicted_battery < critical_battery:
        warning = True
        reason = "Battery level critically low"

    elif not reachable:
        warning = True
        reason = "No reachable charging stations within safe range"

    return AnalysisResponse(
        predicted_range_km=predicted_range_km,
        warning=warning,
        warning_reason=reason,
        recommended_stations=reachable

    )
  """
import time

@app.post("/simulate-drive")
def simulate_drive(
    duration_sec: int = 120,
    start_soc: float = 0.40
):
    """
    Simulate real-time EV driving and Guardian behavior
    """
    soc = start_soc
    results = []

    for t in range(duration_sec):

        # realistic speed pattern
        if t < 30:
            speed = 40
        elif t < 70:
            speed = 60
        else:
            speed = 80

        # simulate battery current draw
        battery_current = speed * 1.2

        status = brain.step(
            speed_kmph=speed,
            dt_sec=1,
            current_soc=soc,
            battery_current=battery_current,
            battery_voltage=390,
            battery_temp=25,
            current_lat=17.3850 + t * 0.00001,
            current_lon=78.4867 + t * 0.00001
        )

        # simulate real SOC drain (ground truth)
        soc = max(0.0, soc - (0.00006 * speed))

        # store only important info
        results.append({
            "time_sec": t,
            "speed": speed,
            "current_soc": soc,
            "predicted_soc": status["predicted_soc"],
            "remaining_range_km": status["remaining_range_km"],
            "alert_level": status["alert_level"],
            "warning_message": status["warning_message"],
            "recommended_station_km": status["recommended_station_km"]
        })

        time.sleep(0.05)  # small delay for realism (not blocking too much)

    return results



@app.get("/")
def root():
    return {
        "status": "running",
        "service": "EV Battery Guardian API"
    }
