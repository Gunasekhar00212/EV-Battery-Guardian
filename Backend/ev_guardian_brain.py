from pyexpat import features
import numpy as np
from collections import deque
import pandas as pd

USER_MODES = {
    "eco": {
        "info": 0.40,
        "warning": 0.25,
        "critical": 0.15,
        "emergency": 0.08,
        "safety_margin_km": 15
    },
    "normal": {
        "info": 0.30,
        "warning": 0.20,
        "critical": 0.10,
        "emergency": 0.05,
        "safety_margin_km": 10
    },
    "aggressive": {
        "info": 0.20,
        "warning": 0.15,
        "critical": 0.07,
        "emergency": 0.03,
        "safety_margin_km": 5
    }
}


class EVGuardianBrain:

    
    def __init__(
        self,
        model,
        scaler,
        charging_stations_km=None,
        user_mode="normal",
        window_size=20
    ):
        

        # --- ML components ---
        self.model = model
        self.scaler = scaler
        self.window_size = window_size

        # --- User mode configuration ---
        if user_mode not in USER_MODES:
            raise ValueError(f"Invalid user_mode: {user_mode}")

        self.user_mode = user_mode
        self.mode_cfg = USER_MODES[user_mode]

        # --- Route / station knowledge ---
        self.charging_stations_km = charging_stations_km or []
        self.safety_margin_km = self.mode_cfg["safety_margin_km"]

        # --- Internal state (memory) ---
        self.feature_window = deque(maxlen=window_size)
        self.distance_travelled_km = 0.0

        # --- Runtime state ---
        self.current_soc = None
        self.predicted_soc = None
        self.remaining_range_km = None
        self.alert_level = "SAFE"
        self.alert_message = ""
        # --- Recommendation state ---
        self.recommended_station_km = None
        self.recommendation_active = False



    # -------------------------------------------------

    def step(
        self,
        speed_kmph,
        dt_sec,
        current_soc,
        battery_current,
        battery_voltage,
        battery_temp
    ):
        """
        One brain update step (e.g., 1 second)
        """

        # 1️⃣ Update distance
        distance_step = (speed_kmph * dt_sec) / 3600.0
        self.distance_travelled_km += distance_step

        self.current_soc = current_soc

        # 2️⃣ Build feature vector (ORDER MUST MATCH TRAINING)
        features = [
            speed_kmph,
            battery_current,
            battery_voltage,
            battery_temp,
            current_soc
        ]

        df_features = pd.DataFrame(
            [features],
            columns=[
                'Velocity [km/h]',
                'Battery Current [A]',
                'Battery Voltage [V]',
                'Battery Temperature [°C]',
                'SoC [%]'
            ]
        )
        scaled_features = self.scaler.transform(df_features)[0]
        # safety check
        if not np.isfinite(scaled_features).all():
            return self.get_status()
        self.feature_window.append(scaled_features)

        # Wait until window is full
        if len(self.feature_window) < self.window_size:
            return self.get_status()

        # 5️⃣ Predict next SOC (scaled)
        self.predicted_soc = self._predict_soc()

        # 6️⃣ Estimate remaining range
        self.remaining_range_km = self._estimate_range()

        # 7️⃣ Analyze charging stations
        next_station_distance = self._analyze_stations()

        # 8️⃣ Decide alert
        self._decide_alert(next_station_distance)

        return self.get_status()

    # -------------------------------------------------

    def _predict_soc(self):
        # 🚨 Guard 1: SOC too low → stop predicting
        if self.current_soc is None or self.current_soc < 0.05:
            return self.current_soc
    
        X = np.array(self.feature_window, dtype=np.float32)
    
        # 🚨 Guard 2: window sanity check
        if X.shape != (self.window_size, 5):
            return self.current_soc
    
        if not np.isfinite(X).all():
            return self.current_soc
    
        X = X.reshape(1, self.window_size, 5)
    
        try:
            predicted_soc_scaled = self.model.predict(X, verbose=0)[0][0]
        except Exception:
            # fail-safe: never crash the system
            return self.current_soc
    
        # inverse scale SOC only
        soc_min = self.scaler.data_min_[-1]
        soc_max = self.scaler.data_max_[-1]
    
        predicted_soc = predicted_soc_scaled * (soc_max - soc_min) + soc_min
    
        return float(np.clip(predicted_soc, 0.0, 1.0))
    

    # -------------------------------------------------

    def _estimate_range(self):
        """
        Estimate remaining range from SOC trend
        """

        soc_drop = self.current_soc - self.predicted_soc

        if soc_drop <= 0:
            return float("inf")

        # SOC drop per km (using recent movement)
        drain_per_km = soc_drop / max(
            (self.distance_travelled_km / max(len(self.feature_window), 1)),
            0.001
        )

        remaining_range = self.current_soc / drain_per_km
        return max(0.0, remaining_range)

    # -------------------------------------------------

    def _analyze_stations(self):
        """
        Route-based station analysis
        """

        for station_km in self.charging_stations_km:
            if station_km > self.distance_travelled_km:
                return station_km - self.distance_travelled_km

        return None

    # -------------------------------------------------

    def _decide_alert(self, next_station_distance):
        soc = self.current_soc
        cfg = self.mode_cfg

        # estimate time-to-empty (minutes)
        if self.remaining_range_km not in [None, float("inf")]:
            avg_speed = max(10, np.mean([f[0] for f in self.feature_window]))
            time_to_empty = (self.remaining_range_km / avg_speed) * 60
        else:
            time_to_empty = None

        # ---------------- EMERGENCY ----------------
        if soc <= cfg["emergency"] or (time_to_empty is not None and time_to_empty < 2):
            self.alert_level = "EMERGENCY"
            self.alert_message = (
                "Battery critically low. Vehicle may stop any moment. "
                "Charge immediately."
            )
            return

        # ---------------- CRITICAL ----------------
        if soc <= cfg["critical"]:
            self.alert_level = "CRITICAL"
            self.alert_message = (
                "Battery very low. Reduce speed and charge immediately."
            )
            return

        if (
            next_station_distance is not None
            and self.remaining_range_km not in [None, float("inf")]
            and self.remaining_range_km < next_station_distance + self.safety_margin_km
        ):
            self.alert_level = "CRITICAL"
            self.alert_message = (
                "Insufficient charge to safely reach next station. "
                "Immediate action required."
            )
            return

        # ---------------- WARNING ----------------
        if soc <= cfg["warning"]:
            self.alert_level = "WARNING"
            self.alert_message = (
                "Battery dropping. Plan charging soon."
            )
            return

        # ---------------- INFO ----------------
        if soc <= cfg["info"]:
            self.alert_level = "INFO"
            self.alert_message = (
                "Battery below normal level. Consider charging."
            )
            return

        # ---------------- SAFE ----------------
        self.alert_level = "SAFE"
        self.alert_message = "Battery level normal. Driving conditions safe."

    def get_status(self):
        return {
            "current_soc": self.current_soc,
            "predicted_soc": self.predicted_soc,
            "remaining_range_km": self.remaining_range_km,
            "distance_travelled_km": self.distance_travelled_km,
            "alert_level": self.alert_level,
            "alert_message": self.alert_message,
            "recommended_station_km": self.recommended_station_km,
            "recommendation_active": self.recommendation_active
        }
