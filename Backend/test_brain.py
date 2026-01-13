from tensorflow.keras.models import load_model
import joblib
import time

from ev_guardian_brain import EVGuardianBrain

# -------------------------------------------------
# 1️⃣ Load trained model & scaler (inference only)
# -------------------------------------------------
model = load_model("models/ev_soc_lstm_model.keras", compile=False)
scaler = joblib.load("models/ev_feature_scaler.pkl")

# -------------------------------------------------
# 2️⃣ Initialize EV Guardian Brain
# -------------------------------------------------
brain = EVGuardianBrain(
    model=model,
    scaler=scaler,
    charging_stations_km=[20, 55, 90],
    user_mode="eco"        # try: eco / normal / aggressive
)


print("EV Guardian Brain initialized successfully")

# -------------------------------------------------
# 3️⃣ Simulation parameters
# -------------------------------------------------
speed_profile = [
    (40, 30),   # 40 km/h for 30 seconds
    (60, 60),   # 60 km/h for 60 seconds
    (80, 90),   # 80 km/h for 90 seconds
]

current_soc = 0.35          # 35% SOC
battery_voltage = 390       # constant for simulation
battery_temp = 25           # constant for simulation

print("\n--- Starting EV Driving Simulation ---\n")

t = 0
dt = 1  # 1 second timestep

# -------------------------------------------------
# 4️⃣ Main simulation loop
# -------------------------------------------------
for speed_kmph, duration in speed_profile:
    for _ in range(duration):

        # simulate battery current draw
        battery_current = speed_kmph * 1.2

        # brain perception + prediction
        status = brain.step(
            speed_kmph=speed_kmph,
            dt_sec=dt,
            current_soc=current_soc,
            battery_current=battery_current,
            battery_voltage=battery_voltage,
            battery_temp=battery_temp
        )

        # -------------------------------------------------
        # REALISTIC SOC UPDATE (ground truth simulation)
        # -------------------------------------------------
        soc_drain = 0.00005 * speed_kmph
        current_soc = max(0.0, current_soc - soc_drain)

        # -------------------------------------------------
        # Print every 5 seconds (safe formatting)
        # -------------------------------------------------
        if t % 5 == 0:
            soc = status["current_soc"]
            rng = status["remaining_range_km"]

            soc_str = f"{soc:.3f}" if soc is not None else "N/A"
            rng_str = f"{rng:.1f}" if rng is not None else "N/A"

            print(
                f"t={t:4d}s | "
                f"Speed={speed_kmph:3d} km/h | "
                f"SOC={soc_str} | "
                f"Range={rng_str} km | "
                f"Alert={status['alert_level']}"
            )

        t += dt
        time.sleep(0.05)

print("\n--- Simulation ended ---")
