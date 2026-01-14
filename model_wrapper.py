#!/usr/bin/env python3
"""
model_wrapper.py

Wrapper class for maize yield prediction model exposing:
- describe_features(): JSON schema of all input features
- predict_response_curve(): nitrogen dose-response curve prediction
"""

import os
import sys
from pathlib import Path

# Add maisN to Python path to access custom layers
maisn_path = Path(__file__).resolve().parent.parent / "maisN"
if maisn_path.exists():
    sys.path.insert(0, str(maisn_path))

# CRITICAL: Set Keras backend BEFORE importing keras
os.environ["KERAS_BACKEND"] = "torch"

import pickle
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import keras
import toml

# Import custom layers for model loading
try:
    from lib.model.fusion_layers import (
        GatedFusionLayer,
        CrossAttentionFusion,
        HierarchicalFusion,
        BiTemporalFusion,
    )
    from lib.model.layers import (
        ConcaveLayer,
        InputConcaveNN,
        ConcavityRegularizer,
        QuadraticDoseResponse,
        MitscherlichDoseResponse,
        LinearPlateauDoseResponse,
        QuadraticPlateauDoseResponse,
    )
except ImportError as e:
    print(f"Warning: Could not import custom layers: {e}")
    print(f"Make sure maisN project is at: {maisn_path}")


def compute_soil_ilr(
    clay_pct: float, sand_pct: float, silt_pct: Optional[float] = None
) -> tuple:
    """
    Compute soil texture ILR (Isometric Log Ratio) from clay, sand, silt percentages.

    Args:
        clay_pct: Clay percentage (0-100)
        sand_pct: Sand percentage (0-100)
        silt_pct: Silt percentage (0-100). If None, computed as 100 - clay - sand

    Returns:
        (ilr1, ilr2): ILR transformed soil texture coordinates
    """
    # Compute silt if not provided
    if silt_pct is None:
        silt_pct = 100.0 - clay_pct - sand_pct

    # Replace zeros with small constant
    zero_replace = 0.01
    clay = max(clay_pct, zero_replace)
    sand = max(sand_pct, zero_replace)
    silt = max(silt_pct, zero_replace)

    # ILR transformation (Egozcue et al. 2003)
    ilr1 = np.sqrt(1 / 2) * np.log(clay / sand)
    ilr2 = np.sqrt(2 / 3) * np.log(silt / np.sqrt(clay * sand))

    # Handle inf/nan
    if not np.isfinite(ilr1):
        ilr1 = 0.0
    if not np.isfinite(ilr2):
        ilr2 = 0.0

    return ilr1, ilr2


def normalize_seeding_density(density_plants_ha: float) -> float:
    """
    Normalize seeding density around typical Quebec maize density.
    
    NOTE: This is a feature engineering transformation that happens BEFORE
    the preprocessor scaler. The preprocessor scaler will then scale this
    normalized value along with other features.
    
    Parameters match config.toml [FEATURES.SEEDING_DENSITY]:
    - NORMALIZE_CENTER = 77500
    - NORMALIZE_SCALE = 10000

    Args:
        density_plants_ha: Seeding density in plants/ha

    Returns:
        Normalized density: (density - 77500) / 10000
    """
    center = 77500.0
    scale = 10000.0
    return (density_plants_ha - center) / scale


def normalize_hybrid_utm(utm_raw: float) -> float:
    """
    Normalize hybrid maturity (UTM) around typical value.
    
    NOTE: This is a feature engineering transformation that happens BEFORE
    the preprocessor scaler. The preprocessor scaler will then scale this
    normalized value along with other features.
    
    Parameters match config.toml [FEATURES.HYBRID]:
    - NORMALIZE_CENTER = 2800
    - NORMALIZE_SCALE = 200

    Args:
        utm_raw: Raw UTM value (Unités Thermiques Maïs)

    Returns:
        Normalized UTM: (utm - 2800) / 200
    """
    center = 2800.0
    scale = 200.0
    return (utm_raw - center) / scale


def transform_no3n(no3n_kg_ha: float) -> float:
    """
    Transform soil NO3-N to log scale.
    
    NOTE: This is a feature engineering transformation that happens BEFORE
    the preprocessor scaler. The preprocessor scaler will then scale this
    log-transformed value along with other features.
    
    Parameters match config.toml [FEATURES.SOIL_NO3N]:
    - LOG_OFFSET = 0.01

    Args:
        no3n_kg_ha: Soil NO3-N in kg N/ha

    Returns:
        Log-transformed NO3-N: log(NO3N + 0.01)
    """
    log_offset = 0.01
    return np.log(max(no3n_kg_ha, 0.0) + log_offset)


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
        # Load model with custom objects
        keras.config.enable_unsafe_deserialization()
        custom_objects = {
            "GatedFusionLayer": GatedFusionLayer,
            "CrossAttentionFusion": CrossAttentionFusion,
            "HierarchicalFusion": HierarchicalFusion,
            "BiTemporalFusion": BiTemporalFusion,
            "ConcaveLayer": ConcaveLayer,
            "InputConcaveNN": InputConcaveNN,
            "ConcavityRegularizer": ConcavityRegularizer,
            "QuadraticDoseResponse": QuadraticDoseResponse,
            "MitscherlichDoseResponse": MitscherlichDoseResponse,
            "LinearPlateauDoseResponse": LinearPlateauDoseResponse,
            "QuadraticPlateauDoseResponse": QuadraticPlateauDoseResponse,
            "ops": keras.ops,
        }

        self.model = keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False, safe_mode=False
        )
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        # Load hyperparameters from training_results file
        self.hyperparams = self._load_hyperparameters(model_path)

        # Check if model is physics-aware (needs unscaled N dose)
        # NOTE: icnn is physics-aware but needs normalized [0,1] doses, not raw kg/ha
        physics_strategies_raw_dose = {
            "quadratic", "mitscherlich", "linear_plateau", "quadratic_plateau",
            "combined"
        }
        physics_strategies_normalized_dose = {"icnn"}
        
        self.dose_response_strategy = self.hyperparams.get("dose_response_strategy", "")
        self.needs_raw_dose = self.dose_response_strategy in physics_strategies_raw_dose
        self.needs_normalized_dose = self.dose_response_strategy in physics_strategies_normalized_dose
        self.is_physics_aware = self.needs_raw_dose or self.needs_normalized_dose

        if self.needs_raw_dose:
            print(f"Model is physics-aware ({self.dose_response_strategy}): N dose will be UNSCALED (raw kg/ha) before prediction")
        elif self.needs_normalized_dose:
            print(f"Model is physics-aware ({self.dose_response_strategy}): N dose will be NORMALIZED [0,1] before prediction")
        else:
            print(f"Model uses standard scaling: dose_response_strategy={self.dose_response_strategy}")
            if not self.dose_response_strategy:
                print("  WARNING: Could not determine dose_response_strategy from hyperparameters!")
                print("  This may cause incorrect predictions. Check if training_results file exists.")

        # Store feature metadata
        self.static_features = self._get_static_feature_metadata()
        self.weather_features = self._get_weather_feature_metadata()

    def _load_hyperparameters(self, model_path: str) -> Dict:
        """
        Load hyperparameters from training_results_{timestamp}.toml file.
        
        Args:
            model_path: Path to .keras model file
            
        Returns:
            Dictionary of hyperparameters, empty dict if not found
        """
        try:
            model_file = Path(model_path)
            # Extract timestamp from model filename (e.g., final_model_20260112_212807.keras)
            if "final_model_" in model_file.stem:
                timestamp = model_file.stem.replace("final_model_", "")
            elif "baseline_model" in model_file.stem:
                # For baseline models, try to find any training_results file
                timestamp = None
            else:
                # Try to extract timestamp pattern YYYYMMDD_HHMMSS
                import re
                match = re.search(r"(\d{8}_\d{6})", model_file.stem)
                if match:
                    timestamp = match.group(1)
                else:
                    timestamp = None
            
            if timestamp:
                # Look for training_results file in same directory as model
                results_path = model_file.parent / f"training_results_{timestamp}.toml"
                if results_path.exists():
                    metadata = toml.load(results_path)
                    hyperparams = metadata.get("hyperparameters", {})
                    print(f"✓ Loaded hyperparameters from: {results_path.name}")
                    print(f"  dose_response_strategy: {hyperparams.get('dose_response_strategy', 'NOT FOUND')}")
                    return hyperparams
                else:
                    # Also check in maisN data/model_results directory
                    maisn_results_path = maisn_path / "data" / "model_results" / f"training_results_{timestamp}.toml"
                    if maisn_results_path.exists():
                        metadata = toml.load(maisn_results_path)
                        hyperparams = metadata.get("hyperparameters", {})
                        print(f"✓ Loaded hyperparameters from: {maisn_results_path}")
                        print(f"  dose_response_strategy: {hyperparams.get('dose_response_strategy', 'NOT FOUND')}")
                        return hyperparams
                    else:
                        print(f"⚠ Warning: Training results file not found:")
                        print(f"   1. {results_path}")
                        print(f"   2. {maisn_results_path}")
                        print(f"  Attempted to load hyperparameters for timestamp: {timestamp}")
                        print(f"  Model will default to standard scaling (may be incorrect for physics-aware models!)")
        except Exception as e:
            print(f"Warning: Could not load hyperparameters: {e}")
        
        return {}

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
                "default": 46.5,
            },
            "longitude": {
                "label": "Longitude",
                "description": "Coordonnée géographique (longitude)",
                "type": "numeric",
                "unit": "degrés",
                "min": -75.0,
                "max": -65.0,
                "default": -71.5,
            },
            "ph_eau": {
                "label": "pH du sol",
                "description": "pH du sol mesuré à l'eau",
                "type": "numeric",
                "unit": "pH",
                "min": 4.5,
                "max": 8.5,
                "default": 6.5,
            },
            "som_log_ratio": {
                "label": "Matière organique",
                "description": "Log-ratio de la matière organique du sol",
                "type": "numeric",
                "unit": "log(%)",
                "min": -5.0,
                "max": -2.0,
                "default": -3.4,
            },
            "clay_pct": {
                "label": "Argile",
                "description": "Pourcentage d'argile dans le sol (0-100%)",
                "type": "numeric",
                "unit": "%",
                "min": 0.0,
                "max": 100.0,
                "default": 25.0,
            },
            "sand_pct": {
                "label": "Sable",
                "description": "Pourcentage de sable dans le sol (0-100%). Le limon sera calculé automatiquement.",
                "type": "numeric",
                "unit": "%",
                "min": 0.0,
                "max": 100.0,
                "default": 35.0,
            },
            "tillage_ord": {
                "label": "Type de travail du sol",
                "description": "Intensité du travail du sol",
                "type": "categorical",
                "options": [
                    {"value": 0, "label": "Labour conventionnel"},
                    {"value": 1, "label": "Labour réduit"},
                    {"value": 2, "label": "Semis direct"},
                ],
                "default": 1,
            },
            "prev_crop_n": {
                "label": "Culture précédente",
                "description": "Type de culture de l'année précédente (effet sur N résiduel)",
                "type": "categorical",
                "options": [
                    {"value": 0, "label": "Maïs (faible N résiduel)"},
                    {"value": 1, "label": "Céréale / Autre (N moyen)"},
                    {"value": 2, "label": "Prairie / Fourrage (N moyen-élevé)"},
                    {"value": 3, "label": "Légumineuse (N élevé)"},
                ],
                "default": 1,
            },
            "density_plants_ha": {
                "label": "Densité de semis",
                "description": "Nombre de plants par hectare",
                "type": "numeric",
                "unit": "plants/ha",
                "min": 60000,
                "max": 95000,
                "default": 77500,
            },
            "annee": {
                "label": "Année",
                "description": "Année de culture",
                "type": "numeric",
                "unit": "année",
                "min": 1990,
                "max": 2030,
                "default": 2024,
            },
            "semis_jd": {
                "label": "Date de semis (jour julien)",
                "description": "Jour julien du semis (1-366, où 1 = 1er janvier)",
                "type": "numeric",
                "unit": "jour",
                "min": 1,
                "max": 366,
                "default": 132,  # Typical: May 12
            },
            "recolte_jd": {
                "label": "Date de récolte (jour julien)",
                "description": "Jour julien de la récolte (1-366)",
                "type": "numeric",
                "unit": "jour",
                "min": 1,
                "max": 366,
                "default": 280,  # Typical: October 7
            },
            "hybrid_utm_norm": {
                "label": "Maturité hybride (UTM)",
                "description": "Unités Thermiques Maïs - normalisé autour de 2800",
                "type": "numeric",
                "unit": "UTM",
                "min": 2400,
                "max": 3200,
                "default": 2800,
            },
            "no3n_log": {
                "label": "Azote nitrique du sol (NO3-N)",
                "description": "Azote nitrique au semis (kg N/ha), transformé en log",
                "type": "numeric",
                "unit": "log(kg N/ha)",
                "min": -5.0,
                "max": 5.0,
                "default": 2.0,  # ~7.4 kg N/ha
            },
            "no3n_was_imputed": {
                "label": "NO3-N imputé",
                "description": "Indicateur si la valeur NO3-N a été imputée (0=mesuré, 1=imputé)",
                "type": "categorical",
                "options": [
                    {"value": 0, "label": "Mesuré"},
                    {"value": 1, "label": "Imputé"},
                ],
                "default": 0,
            },
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
            features.append(
                {
                    "name": col,
                    "label": weather_labels.get(col, col),
                    "description": f"Série temporelle journalière: {col}",
                    "type": "time_series",
                    "unit": self._infer_unit(col),
                    "required_days_pre": self.preprocessor.pre_seedling_days,
                    "required_days_growing": self.preprocessor.max_growing_days,
                }
            )

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
                "time_series": self.weather_features,
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
                "step": 10,
            },
            "output": {
                "name": "yield",
                "label": "Rendement",
                "description": "Rendement prédit du maïs grain",
                "unit": "t/ha",
            },
            "preprocessing": {
                "pre_seedling_days": self.preprocessor.pre_seedling_days,
                "max_growing_days": self.preprocessor.max_growing_days,
                "n_static_features": len(self.preprocessor.kept_static_cols),
                "n_weather_features": len(self.preprocessor.weather_cols),
            },
        }

    def predict_response_curve(
        self,
        features: Dict,
        nitrogen_range: Optional[List[float]] = None,
        weather_data: Optional[Dict[str, pd.DataFrame]] = None,
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
            pre_weather = self._prepare_weather_from_df(
                weather_data.get("pre_seedling")
            )
            growing_weather = self._prepare_weather_from_df(weather_data.get("growing"))
        else:
            # Use default/representative weather pattern
            pre_weather = np.zeros(
                (
                    1,
                    self.preprocessor.pre_seedling_days,
                    len(self.preprocessor.weather_cols),
                )
            )
            growing_weather = np.zeros(
                (
                    1,
                    self.preprocessor.max_growing_days,
                    len(self.preprocessor.weather_cols),
                )
            )

        # Tile weather data for all doses
        pre_weather_batch = np.tile(pre_weather, (n_doses, 1, 1))
        growing_weather_batch = np.tile(growing_weather, (n_doses, 1, 1))

        # Prepare nitrogen doses
        n_dose_array = np.array(nitrogen_range).reshape(-1, 1)
        
        # CRITICAL: Handle N dose scaling based on model type
        # - Physics-aware models (quadratic, mitscherlich, etc.): need RAW doses (kg/ha)
        # - ICNN models: need NORMALIZED [0,1] doses
        # - Standard models: need SCALED doses (using preprocessor scaler)
        if self.needs_raw_dose:
            # For physics-aware models (except icnn), use unscaled doses directly
            n_dose_for_model = n_dose_array.astype(np.float32).flatten()
        elif self.needs_normalized_dose:
            # For icnn, normalize to [0,1] range
            # Get the scaler's min/max to normalize properly
            scaler = self.preprocessor.scalers["n_dose"]
            if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                # MinMaxScaler
                min_val = scaler.data_min_[0]
                max_val = scaler.data_max_[0]
                n_dose_for_model = ((n_dose_array - min_val) / (max_val - min_val)).astype(np.float32).flatten()
            else:
                # StandardScaler or other - use transform then normalize
                scaled = scaler.transform(n_dose_array)
                # Normalize scaled values to [0,1]
                scaled_min, scaled_max = scaled.min(), scaled.max()
                if scaled_max > scaled_min:
                    n_dose_for_model = ((scaled - scaled_min) / (scaled_max - scaled_min)).astype(np.float32).flatten()
                else:
                    n_dose_for_model = scaled.astype(np.float32).flatten()
        else:
            # For standard models, scale the doses using preprocessor scaler
            n_dose_scaled = self.preprocessor.scalers["n_dose"].transform(n_dose_array)
            n_dose_for_model = n_dose_scaled.astype(np.float32).flatten()

        # Debug output for first few doses
        if len(nitrogen_range) > 0:
            print(f"DEBUG: N dose scaling check (first 3 doses):")
            print(f"  Input range: {nitrogen_range[:3]} kg/ha")
            print(f"  Model input: {n_dose_for_model[:3]}")
            print(f"  Strategy: {self.dose_response_strategy}, needs_raw={self.needs_raw_dose}, needs_norm={self.needs_normalized_dose}")

        # Build input dictionary
        X = {"n_dose": n_dose_for_model, "static_features": static_features}

        # Add weather inputs if model expects them
        if (
            self.preprocessor.pre_seedling_days > 0
            and len(self.preprocessor.weather_cols) > 0
        ):
            X["pre_seedling_weather"] = pre_weather_batch
        if (
            self.preprocessor.max_growing_days > 0
            and len(self.preprocessor.weather_cols) > 0
        ):
            X["growing_weather"] = growing_weather_batch

        # Predict (model outputs raw yield values, not scaled)
        predictions = self.model.predict(X, verbose=0).flatten()

        # Return as DataFrame
        return pd.DataFrame({"n_dose": nitrogen_range, "predicted_yield": predictions})

    def _prepare_static_features(self, features: Dict) -> np.ndarray:
        """
        Convert feature dictionary to scaled numpy array.
        Transforms user-friendly inputs (clay_pct, sand_pct, density_plants_ha)
        to model features (soil_ilr1, soil_ilr2, density_norm).

        Args:
            features: Dict with feature name -> value (can include raw inputs)

        Returns:
            Scaled feature array (1, n_features)
        """
        # Transform raw inputs to model features
        transformed_features = features.copy()

        # Transform soil texture to ILR if raw percentages provided
        if "clay_pct" in features and "sand_pct" in features:
            ilr1, ilr2 = compute_soil_ilr(features["clay_pct"], features["sand_pct"])
            transformed_features["soil_ilr1"] = ilr1
            transformed_features["soil_ilr2"] = ilr2
            # Remove raw inputs
            transformed_features.pop("clay_pct", None)
            transformed_features.pop("sand_pct", None)

        # Transform seeding density to normalized form
        if "density_plants_ha" in features:
            transformed_features["density_norm"] = normalize_seeding_density(
                features["density_plants_ha"]
            )
            # Remove raw input
            transformed_features.pop("density_plants_ha", None)

        # Transform hybrid UTM to normalized form
        if "hybrid_utm_raw" in features:
            transformed_features["hybrid_utm_norm"] = normalize_hybrid_utm(
                features["hybrid_utm_raw"]
            )
            # Remove raw input
            transformed_features.pop("hybrid_utm_raw", None)

        # Transform soil NO3-N to log scale
        if "no3n_kg_ha" in features and features["no3n_kg_ha"] is not None:
            transformed_features["no3n_log"] = transform_no3n(
                features["no3n_kg_ha"]
            )
            # Remove raw input
            transformed_features.pop("no3n_kg_ha", None)
            # Set imputation flag if not provided
            if "no3n_was_imputed" not in transformed_features:
                transformed_features["no3n_was_imputed"] = 0  # Assume measured
        elif "no3n_was_imputed" in features:
            # User indicated NO3N is imputed but didn't provide value
            # Use imputed value from preprocessor
            transformed_features["no3n_log"] = self.preprocessor.static_impute_values.get("no3n_log", 2.0)
            transformed_features["no3n_was_imputed"] = features["no3n_was_imputed"]

        # Create array with correct order matching preprocessor's kept_static_cols
        # This ensures features are in the same order as when the scaler was fitted
        feature_values = []
        for col in self.preprocessor.kept_static_cols:
            # Get value from transformed features, or use preprocessor's impute value if missing
            value = transformed_features.get(
                col, self.preprocessor.static_impute_values.get(col, 0.0)
            )
            feature_values.append(value)

        # Convert to numpy array (1, n_features)
        feature_array = np.array(feature_values).reshape(1, -1)
        
        # CRITICAL: Scale using the preprocessor's fitted scaler
        # This scaler was fitted on training data with the same feature transformations
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
        grain_price: float = 0.20,
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
            "net_revenue_per_ha": float(optimal["net_revenue"]),
        }
