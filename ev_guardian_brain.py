from pyexpat import features
import numpy as np
from collections import deque
import pandas as pd


class EVGuardianBrain:
    def __init__(
        self,
        model,
        scaler,
        charging_stations_km=None,
        safety_margin_km=10,
        window_size=20
    ):
        """
        model  : trained LSTM model
        scaler : MinMaxScaler used during training (same object)
        """

        # --- ML components ---
        self.model = model
        self.scaler = scaler
        self.window_size = window_size

        # --- Route / station knowledge ---
        self.charging_stations_km = charging_stations_km or []
        self.safety_margin_km = safety_margin_km

        # --- Internal state (memory) ---
        self.feature_window = deque(maxlen=window_size)
        self.distance_travelled_km = 0.0

        self.current_soc = None
        self.predicted_soc = None
        self.remaining_range_km = None

        self.alert_level = "SAFE"
        self.alert_message = "Initializing system"

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
        """
        Generate actionable alert
        """

        if next_station_distance is None:
            self.alert_level = "WARNING"
            self.alert_message = (
                "No upcoming charging stations detected. "
                "Drive conservatively."
            )
            return

        if self.remaining_range_km > next_station_distance + self.safety_margin_km:
            self.alert_level = "SAFE"
            self.alert_message = (
                f"Sufficient charge to reach next station "
                f"({next_station_distance:.1f} km away)."
            )

        elif self.remaining_range_km > next_station_distance:
            self.alert_level = "WARNING"
            self.alert_message = (
                f"Recharge recommended at next station "
                f"({next_station_distance:.1f} km)."
            )

        else:
            self.alert_level = "CRITICAL"
            self.alert_message = (
                f"Recharge immediately. "
                f"Current range insufficient to reach next station "
                f"({next_station_distance:.1f} km)."
            )

    # -------------------------------------------------

    def get_status(self):
        """
        Brain output (API / UI reads ONLY this)
        """
        return {
            "current_soc": self.current_soc,
            "predicted_soc": self.predicted_soc,
            "remaining_range_km": self.remaining_range_km,
            "distance_travelled_km": self.distance_travelled_km,
            "alert_level": self.alert_level,
            "alert_message": self.alert_message
        }
