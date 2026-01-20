import numpy as np
from collections import deque
import pandas as pd
from backend.utils.geo import haversine_distance
from backend.utils.geo import estimate_road_distance_km
from backend.utils.charging_api import fetch_nearby_charging_stations

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
        charging_stations=None,
        user_mode="normal",
        window_size=20,
        safety_margin_km=10
        
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
        self.charging_stations= charging_stations or []
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
        # # --- Recommendation state ---
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
        battery_temp,
        current_lat=None,
        current_lon=None
    ):
        self.current_lat=current_lat
        self.current_lon=current_lon

        self.current_soc = current_soc
        
        


        """
        One brain update step (e.g., 1 second)
        """
        self.full_battery_range_km = 300.0  # realistic default EV range

        # 1️⃣ Update distance
        distance_step = (speed_kmph * dt_sec) / 3600.0
        self.distance_travelled_km += distance_step

        

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
        if df_features.shape[1] != self.scaler.n_features_in_:
            return self.get_status()
        # 3️⃣ Scale features
        scaled_features = self.scaler.transform(df_features.values)[0]
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
        station, distance_km = self._find_reachable_station()

        self.recommended_station_km = distance_km
        self.recommendation_active = station is not None
        self.recommended_station = station



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
            pred = self.model.predict(X, verbose=0)
    
            if pred is None:    
                return self.current_soc
    
            predicted_soc_scaled = float(np.ravel(pred)[0])
        except Exception:
            # fail-safe: never crash the system
            return self.current_soc
    
        # inverse scale SOC only
        soc_min = self.scaler.data_min_[-1]
        soc_max = self.scaler.data_max_[-1]
    
        predicted_soc = predicted_soc_scaled * (soc_max - soc_min) + soc_min
        predicted_soc = float(np.clip(predicted_soc, 0.0, 1.0))
    
        # 🔒 Physics-aware soft constraint
        predicted_soc = min(predicted_soc, self.current_soc - 0.0001)
    
        return predicted_soc

    

    # -------------------------------------------------

    def _estimate_range(self):
        try:
            if self.predicted_soc is None or self.current_soc is None:
                return None

            soc_drop = self.current_soc - self.predicted_soc

            if soc_drop <= 0:
                return None

            if self.distance_travelled_km <= 0:
                return None

            drain_per_km = soc_drop / self.distance_travelled_km

            if not np.isfinite(drain_per_km) or drain_per_km <= 0:
                return None

            remaining_range = self.current_soc / drain_per_km

            if not np.isfinite(remaining_range):
                return None

            return float(max(0.0, remaining_range))

        except Exception:
            return None


    # -------------------------------------------------

    def _analyze_stations(self):
        """
        Fetch real charging stations and find nearest distance
        """
        if self.current_lat is None or self.current_lon is None:
            return None

        stations = fetch_nearby_charging_stations(
            self.current_lat,
            self.current_lon,
            radius_km=20
        )

        if not stations:
            return None

        nearest_distance = None

        for station in stations:
            dist = haversine_distance(
                self.current_lat,
                self.current_lon,
                station["lat"],
                station["lon"]
            )

            if nearest_distance is None or dist < nearest_distance:
                nearest_distance = dist

        return nearest_distance
            # -------------------------------------------------

    def _decide_alert(self, next_station_distance):
        soc = self.current_soc
        cfg = self.mode_cfg

        # estimate time-to-empty (minutes)
        if self.remaining_range_km not in [None, float("inf")]:
            try:
                avg_speed = max(10, abs(np.mean([f[0] for f in self.feature_window])))
            except Exception:
                avg_speed = 10

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
    def _generate_warning(self):

        """
            Convert alert level + internal state into human-like warnings
        """

        level = self.alert_level

        if level == "SAFE":
            return {
                "warning_message": "Battery status is healthy. No action needed.",
                "action_required": False,
                "reason": "SOC and predicted range are within safe limits."
            }

        if level == "INFO":
            return {
                "warning_message": "Battery is slowly declining. Stay aware of upcoming charging options.",
                "action_required": False,
                "reason": "SOC trend shows gradual consumption."
            }

        if level == "WARNING":
            return {
                "warning_message": "Battery is dropping faster than expected. Plan to charge soon.",
                "action_required": True,
                "reason": "Predicted range is approaching safety margin."
            }

        if level == "CRITICAL":
            if self.recommended_station_km is not None:
                return {
                    "warning_message": (
                        f"Battery critically low. "
                        f"Charge at the station in "
                        f"{self.recommended_station_km - self.distance_travelled_km:.1f} km."
                    ),
                    "action_required": True,
                    "reason": "This is the nearest reachable charging station."
                }
            else:
                return {
                    "warning_message": (
                        "Battery critically low. No reachable charging stations ahead."
                    ),
                    "action_required": True,
                    "reason": "Remaining range cannot safely reach any known station."
                }


        if level == "EMERGENCY":
            return {
                "warning_message": "Battery almost empty. Stop and charge immediately to avoid being stranded.",
                "action_required": True,
                "reason": "SOC has reached emergency threshold."
            }

        return {
            "warning_message": "Battery status unknown.",
            "action_required": False,
            "reason": "Insufficient data."
        }
    def _find_reachable_station(self):
        if self.current_lat is None or self.current_lon is None:
            return None, None

        remaining_range_km = self._estimate_remaining_range()
        if remaining_range_km is None:
            return None, None

        reachable = []

        for station in self.charging_stations:
            road_distance_km = estimate_road_distance_km(
                self.current_lat,
                self.current_lon,
                station["lat"],
                station["lon"]
            )

            if remaining_range_km >= road_distance_km + self.safety_margin_km:
                reachable.append((station, road_distance_km))

        if not reachable:
            return None, None

        best_station, best_distance = min(reachable, key=lambda x: x[1])
        return best_station, best_distance


        
    # -------------------------------------------------
    def _estimate_remaining_range(self):
        if self.current_soc is None:
            return None
        return round(self.current_soc * self.full_battery_range_km, 2)



    def get_status(self):
        warning = self._generate_warning()

        return {
            "current_soc": self.current_soc,
            "predicted_soc": self.predicted_soc,
            "remaining_range_km": self._estimate_remaining_range(),
            "distance_travelled_km": self.distance_travelled_km,

            "alert_level": self.alert_level,
            "alert_message": self.alert_message,

            "warning_message": warning["warning_message"],
            "action_required": warning["action_required"],
            "reason": warning["reason"],

            "recommended_station_km": self.recommended_station_km,
            "distance_to_station_km": (
                self.recommended_station_km - self.distance_travelled_km
                if self.recommended_station_km is not None else None
            )
        }

