#!/usr/bin/env python3
"""
model_wrapper.py

Wrapper class for maize yield prediction model exposing:
- describe_features(): JSON schema of all input features
- predict_response_curve(): nitrogen dose-response curve prediction
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import keras


class MaizeYieldPredictor:
    """
    Wrapper for trained maize yield prediction model.

    Provides a clean API for:
    - Feature schema inspection
    - Nitrogen dose-response curve prediction
    """

    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Load trained model and preprocessor.

        Args:
            model_path: Path to .keras model file
            preprocessor_path: Path to .pkl preprocessor file
        """
        self.model = keras.models.load_model(model_path)
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        # Store feature metadata
        self.static_features = self._get_static_feature_metadata()
        self.weather_features = self._get_weather_feature_metadata()

    def _get_static_feature_metadata(self) -> List[Dict]:
        """
        Get metadata for static features with French labels and descriptions.
        """
        # Map technical names to French display names and metadata
        feature_metadata = {
            "latitude": {
                "label": "Latitude",
                "description": "Coordonnée géographique (latitude)",
                "type": "numeric",
                "unit": "degrés",
                "min": 44.0,
                "max": 50.0,
                "default": 46.5
            },
            "longitude": {
                "label": "Longitude",
                "description": "Coordonnée géographique (longitude)",
                "type": "numeric",
                "unit": "degrés",
                "min": -75.0,
                "max": -65.0,
                "default": -71.5
            },
            "ph_eau": {
                "label": "pH du sol",
                "description": "pH du sol mesuré à l'eau",
                "type": "numeric",
                "unit": "pH",
                "min": 4.5,
                "max": 8.5,
                "default": 6.5
            },
            "som_log_ratio": {
                "label": "Matière organique",
                "description": "Log-ratio de la matière organique du sol",
                "type": "numeric",
                "unit": "log(%)",
                "min": -5.0,
                "max": -2.0,
                "default": -3.4
            },
            "soil_ilr1": {
                "label": "Texture du sol (ILR1)",
                "description": "Premier ratio isométrique log de la texture",
                "type": "numeric",
                "unit": "",
                "min": -2.0,
                "max": 2.0,
                "default": 0.0
            },
            "soil_ilr2": {
                "label": "Texture du sol (ILR2)",
                "description": "Deuxième ratio isométrique log de la texture",
                "type": "numeric",
                "unit": "",
                "min": -2.0,
                "max": 2.0,
                "default": 0.0
            },
            "tillage_ord": {
                "label": "Type de travail du sol",
                "description": "Intensité du travail du sol",
                "type": "categorical",
                "options": [
                    {"value": 0, "label": "Labour conventionnel"},
                    {"value": 1, "label": "Labour réduit"},
                    {"value": 2, "label": "Semis direct"}
                ],
                "default": 1
            },
            "prev_crop_n": {
                "label": "Culture précédente",
                "description": "Type de culture de l'année précédente",
                "type": "categorical",
                "options": [
                    {"value": 0, "label": "Légumineuse"},
                    {"value": 1, "label": "Céréale"},
                    {"value": 2, "label": "Prairie"}
                ],
                "default": 1
            },
            "density_norm": {
                "label": "Densité de semis",
                "description": "Densité de semis normalisée (centrée sur 77500 plants/ha)",
                "type": "numeric",
                "unit": "plants/ha (normalisé)",
                "min": -1.5,
                "max": 1.5,
                "default": 0.0
            }
        }

        # Build feature list from preprocessor's kept static columns
        features = []
        for col in self.preprocessor.kept_static_cols:
            if col in feature_metadata:
                meta = feature_metadata[col].copy()
                meta["name"] = col
                features.append(meta)

        return features

    def _get_weather_feature_metadata(self) -> List[Dict]:
        """
        Get metadata for time series weather features.
        """
        # Map weather column names to French labels
        weather_labels = {
            "temperature_2m_mean": "Température moyenne (°C)",
            "temperature_2m_min": "Température minimale (°C)",
            "temperature_2m_max": "Température maximale (°C)",
            "precipitation_sum": "Précipitations (mm)",
            "soil_temperature_0_to_7cm_mean": "Température du sol 0-7cm (°C)",
            "soil_moisture_0_to_7cm_mean": "Humidité du sol 0-7cm (m³/m³)",
            "surface_solar_radiation_downwards_sum": "Radiation solaire (MJ/m²)",
            "wind_speed_10m_mean": "Vitesse du vent (m/s)",
            "potential_evaporation_sum": "Évapotranspiration potentielle (mm)",
        }

        features = []
        for col in self.preprocessor.weather_cols[:10]:  # Top 10 weather features
            features.append({
                "name": col,
                "label": weather_labels.get(col, col),
                "description": f"Série temporelle journalière: {col}",
                "type": "time_series",
                "unit": self._infer_unit(col),
                "required_days_pre": self.preprocessor.pre_seedling_days,
                "required_days_growing": self.preprocessor.max_growing_days
            })

        return features

    def _infer_unit(self, col_name: str) -> str:
        """Infer unit from column name."""
        if "temperature" in col_name.lower():
            return "°C"
        elif "precipitation" in col_name.lower() or "evaporation" in col_name.lower():
            return "mm"
        elif "wind" in col_name.lower():
            return "m/s"
        elif "radiation" in col_name.lower():
            return "MJ/m²"
        elif "moisture" in col_name.lower():
            return "m³/m³"
        else:
            return ""

    def describe_features(self) -> Dict:
        """
        Return JSON schema describing all input features.

        Returns:
            Dictionary with feature schema including:
            - static features (soil, location, management)
            - time series features (weather)
            - nitrogen dose range
        """
        return {
            "model_version": "1.0",
            "description": "Modèle de prédiction du rendement du maïs en fonction de la dose d'azote",
            "features": {
                "static": self.static_features,
                "time_series": self.weather_features
            },
            "nitrogen": {
                "name": "n_dose",
                "label": "Dose d'azote",
                "description": "Quantité d'engrais azoté appliquée",
                "type": "numeric",
                "unit": "kg N/ha",
                "min": 0,
                "max": 300,
                "default_range": [0, 300],
                "step": 10
            },
            "output": {
                "name": "yield",
                "label": "Rendement",
                "description": "Rendement prédit du maïs grain",
                "unit": "t/ha"
            },
            "preprocessing": {
                "pre_seedling_days": self.preprocessor.pre_seedling_days,
                "max_growing_days": self.preprocessor.max_growing_days,
                "n_static_features": len(self.preprocessor.kept_static_cols),
                "n_weather_features": len(self.preprocessor.weather_cols)
            }
        }

    def predict_response_curve(
        self,
        features: Dict,
        nitrogen_range: Optional[List[float]] = None,
        weather_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Predict yield for varying nitrogen doses.

        Args:
            features: Dictionary of static features (soil, location, management)
            nitrogen_range: List of nitrogen doses to test (kg/ha)
                          Default: [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
            weather_data: Optional dict with 'pre_seedling' and 'growing' DataFrames
                         If None, uses default/averaged weather patterns

        Returns:
            DataFrame with columns: n_dose, predicted_yield
        """
        if nitrogen_range is None:
            nitrogen_range = list(range(0, 301, 25))

        n_doses = len(nitrogen_range)

        # Prepare static features
        static_array = self._prepare_static_features(features)
        static_features = np.tile(static_array, (n_doses, 1))

        # Prepare weather features (time series)
        if weather_data is not None:
            pre_weather = self._prepare_weather_from_df(weather_data.get('pre_seedling'))
            growing_weather = self._prepare_weather_from_df(weather_data.get('growing'))
        else:
            # Use default/representative weather pattern
            pre_weather = np.zeros((1, self.preprocessor.pre_seedling_days,
                                   len(self.preprocessor.weather_cols)))
            growing_weather = np.zeros((1, self.preprocessor.max_growing_days,
                                       len(self.preprocessor.weather_cols)))

        # Tile weather data for all doses
        pre_weather_batch = np.tile(pre_weather, (n_doses, 1, 1))
        growing_weather_batch = np.tile(growing_weather, (n_doses, 1, 1))

        # Prepare nitrogen doses
        n_dose_array = np.array(nitrogen_range).reshape(-1, 1)
        n_dose_scaled = self.preprocessor.scalers["n_dose"].transform(n_dose_array)

        # Build input dictionary
        X = {
            "n_dose": n_dose_scaled,
            "static_features": static_features
        }

        # Add weather inputs if model expects them
        if self.preprocessor.pre_seedling_days > 0 and len(self.preprocessor.weather_cols) > 0:
            X["pre_seedling_weather"] = pre_weather_batch
        if self.preprocessor.max_growing_days > 0 and len(self.preprocessor.weather_cols) > 0:
            X["growing_weather"] = growing_weather_batch

        # Predict
        predictions_scaled = self.model.predict(X, verbose=0)

        # Inverse transform predictions
        predictions = self.preprocessor.scalers["yield"].inverse_transform(
            predictions_scaled
        ).flatten()

        # Return as DataFrame
        return pd.DataFrame({
            "n_dose": nitrogen_range,
            "predicted_yield": predictions
        })

    def _prepare_static_features(self, features: Dict) -> np.ndarray:
        """
        Convert feature dictionary to scaled numpy array.

        Args:
            features: Dict with feature name -> value

        Returns:
            Scaled feature array (1, n_features)
        """
        # Create array with correct order
        feature_values = []
        for col in self.preprocessor.kept_static_cols:
            value = features.get(col, self.preprocessor.static_impute_values.get(col, 0.0))
            feature_values.append(value)

        # Convert to numpy and scale
        feature_array = np.array(feature_values).reshape(1, -1)
        feature_scaled = self.preprocessor.scalers["static"].transform(feature_array)

        return feature_scaled

    def _prepare_weather_from_df(self, df: Optional[pd.DataFrame]) -> np.ndarray:
        """
        Convert weather DataFrame to scaled 3D array.

        Args:
            df: DataFrame with columns matching self.preprocessor.weather_cols

        Returns:
            Scaled weather array (1, n_days, n_features)
        """
        if df is None or len(df) == 0:
            # Return zeros (will be filled by model's masking)
            return np.zeros((1, 1, len(self.preprocessor.weather_cols)))

        # Extract weather columns in correct order
        weather_values = df[self.preprocessor.weather_cols].values

        # Scale
        weather_scaled = self.preprocessor.scalers["weather"].transform(weather_values)

        # Add batch dimension
        return weather_scaled.reshape(1, len(weather_scaled), -1)

    def find_optimal_nitrogen(
        self,
        dose_response: pd.DataFrame,
        n_price: float = 1.5,
        grain_price: float = 0.20
    ) -> Dict:
        """
        Find economically optimal nitrogen rate.

        Args:
            dose_response: DataFrame with n_dose, predicted_yield columns
            n_price: Nitrogen fertilizer price ($/kg N)
            grain_price: Grain price ($/kg)

        Returns:
            Dict with optimal_n, max_yield, net_revenue, etc.
        """
        df = dose_response.copy()

        # Calculate economics (yield in t/ha, convert to kg/ha)
        df["revenue"] = df["predicted_yield"] * 1000 * grain_price
        df["cost"] = df["n_dose"] * n_price
        df["net_revenue"] = df["revenue"] - df["cost"]
        df["marginal_revenue"] = df["net_revenue"].diff()

        # Find optimal
        optimal_idx = df["net_revenue"].idxmax()
        optimal = df.loc[optimal_idx]

        return {
            "optimal_n_kg_ha": float(optimal["n_dose"]),
            "predicted_yield_t_ha": float(optimal["predicted_yield"]),
            "revenue_per_ha": float(optimal["revenue"]),
            "cost_per_ha": float(optimal["cost"]),
            "net_revenue_per_ha": float(optimal["net_revenue"])
        }
