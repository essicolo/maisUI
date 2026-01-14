#!/usr/bin/env python3
"""
Maize Yield Prediction - Professional Interface
Clean, minimal, single-page design
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go

# Set Keras backend before any imports
os.environ["KERAS_BACKEND"] = "torch"
sys.path.insert(0, str(Path(__file__).parent.parent / "maisN"))

from model_wrapper import MaizeYieldPredictor

# Global predictor
predictor: Optional[MaizeYieldPredictor] = None


def initialize_model(model_path: str, preprocessor_path: str):
    """Load model and preprocessor."""
    global predictor
    print(f"Loading model: {model_path}")
    print(f"Loading preprocessor: {preprocessor_path}")
    predictor = MaizeYieldPredictor(model_path, preprocessor_path)
    print("Model loaded successfully")


def create_dose_response_plot(
    df: pd.DataFrame, optimal_n: Optional[Dict] = None
) -> go.Figure:
    """Create dose-response curve plot."""
    fig = go.Figure()

    # Main curve
    fig.add_trace(
        go.Scatter(
            x=df["n_dose"],
            y=df["predicted_yield"],
            mode="lines",
            name="Rendement prédit",
            line=dict(color="black", width=2),
            hovertemplate="N: %{x} kg/ha<br>Rendement: %{y:.1f} t/ha<extra></extra>",
        )
    )

    # Optimal point
    if optimal_n:
        fig.add_trace(
            go.Scatter(
                x=[optimal_n["optimal_n_kg_ha"]],
                y=[optimal_n["predicted_yield_t_ha"]],
                mode="markers",
                name="Optimum économique",
                marker=dict(color="black", size=10, symbol="circle"),
                hovertemplate=f"N optimal: {optimal_n['optimal_n_kg_ha']:.0f} kg/ha<br>Rendement: {optimal_n['predicted_yield_t_ha']:.1f} t/ha<extra></extra>",
            )
        )

    # Calculate reference yield for horizontal line (use optimal yield if available, else mean)
    if optimal_n:
        ref_yield = optimal_n["predicted_yield_t_ha"]
    else:
        ref_yield = df["predicted_yield"].mean()
    
    fig.update_layout(
        xaxis_title="Dose d'azote (kg N/ha)",
        yaxis_title="Rendement grain (t/ha)",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,  # Disable legend
        font=dict(family="Arial, sans-serif", size=12, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=30, t=30, b=60)
    )
    
    # Update all traces to hide legend
    fig.update_traces(showlegend=False)

    return fig


def predict_yield(
    # Location
    latitude: float,
    longitude: float,
    # Soil
    clay_pct: float,
    sand_pct: float,
    ph: float,
    organic_matter: float,
    # Management
    prev_crop: int,
    tillage: int,
    density: float,
    hybrid_utm: float,
    # Soil N
    no3n: float,
    no3n_imputed: int,
    # Dates
    year: float,
    seeding_jd: float,
    harvest_jd: float,
    # Nitrogen
    n_min: float,
    n_max: float,
    n_step: float,
    # Weather
    pre_weather_file,
    grow_weather_file,
    # Economics
    n_price: float,
    grain_price: float,
) -> Tuple:
    """Main prediction function."""
    if predictor is None:
        return None, "Erreur: Modèle non chargé"

    try:
        # Load weather data
        if pre_weather_file is None or grow_weather_file is None:
            return None, "Erreur: Veuillez téléverser les deux fichiers météo"

        pre_weather = pd.read_csv(pre_weather_file.name)
        grow_weather = pd.read_csv(grow_weather_file.name)

        # Prepare features
        # Calculate som_log_ratio: log(organic / mineral) as in preprocessing
        mineral_pct = max(100.0 - organic_matter, 1.0)  # clip to avoid log(0)
        som_log_ratio = np.log(organic_matter / mineral_pct)

        features = {
            "latitude": latitude,
            "longitude": longitude,
            "clay_pct": clay_pct,
            "sand_pct": sand_pct,
            "ph_eau": ph,
            "som_log_ratio": som_log_ratio,
            "prev_crop_n": prev_crop,
            "tillage_ord": tillage,
            "density_plants_ha": density,
            "hybrid_utm_raw": float(hybrid_utm),
            "no3n_kg_ha": float(no3n) if no3n is not None else None,
            "no3n_was_imputed": int(no3n_imputed),
            "annee": float(year),
            "semis_jd": float(seeding_jd),
            "recolte_jd": float(harvest_jd),
            "pre_seedling_weather": pre_weather,
            "growing_weather": grow_weather,
        }

        # Generate dose-response curve
        curve = predictor.predict_response_curve(
            features=features, nitrogen_range=np.arange(n_min, n_max + n_step, n_step)
        )

        # Find economic optimum
        optimal = predictor.find_optimal_nitrogen(
            dose_response=curve, n_price=n_price, grain_price=grain_price
        )

        # Create plot
        fig = create_dose_response_plot(curve, optimal)

        # Summary text
        summary = f"""
**Optimum économique**

Dose N optimale: **{optimal["optimal_n_kg_ha"]:.0f} kg N/ha**
Rendement attendu: **{optimal["predicted_yield_t_ha"]:.1f} t/ha**
Revenu: **{optimal["revenue_per_ha"]:.2f} $/ha**
Coût N: **{optimal["cost_per_ha"]:.2f} $/ha**
Revenu net: **{optimal["net_revenue_per_ha"]:.2f} $/ha**
"""

        return fig, summary

    except Exception as e:
        import traceback

        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


def create_interface():
    """Create Gradio interface."""

    # Custom CSS - minimal, professional
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif !important;
        max-width: 1400px !important;
    }
    .input-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
    }
    h1, h2, h3 {
        color: #000 !important;
        font-weight: 600 !important;
    }
    .gr-button {
        background: #000 !important;
        color: #fff !important;
    }
    .gr-button:hover {
        background: #333 !important;
    }
    """

    with gr.Blocks(title="Prédiction de rendement du maïs") as interface:
        gr.Markdown("# Modèle de prédiction de rendement du maïs")
        gr.Markdown(
            "Entrez les caractéristiques du champ, management practices, and weather data to predict nitrogen response."
        )

        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=3):
                gr.Markdown("### Localisation")
                with gr.Row():
                    latitude = gr.Number(
                        label="Latitude (°N)", value=46.5, minimum=44, maximum=50
                    )
                    longitude = gr.Number(
                        label="Longitude (°O)", value=-71.5, minimum=-75, maximum=-65
                    )

                gr.Markdown("### Propriétés du sol")
                with gr.Row():
                    clay = gr.Number(
                        label="Argile (%)", value=25, minimum=0, maximum=100
                    )
                    sand = gr.Number(
                        label="Sable (%)", value=35, minimum=0, maximum=100
                    )
                    ph = gr.Number(label="pH", value=6.5, minimum=4.5, maximum=8.5)
                    som = gr.Number(
                        label="Matière organique (%)",
                        value=3.5,
                        minimum=0.5,
                        maximum=15,
                    )

                gr.Markdown("### Gestion")
                with gr.Row():
                    prev_crop = gr.Dropdown(
                        choices=[
                            ("Maïs", 0),
                            ("Céréale", 1),
                            ("Fourrage", 2),
                            ("Légumineuse", 3),
                        ],
                        value=1,
                        label="Culture précédente",
                    )
                    tillage = gr.Dropdown(
                        choices=[
                            ("Labour conventionnel", 0),
                            ("Labour réduit", 1),
                            ("Semis direct", 2),
                        ],
                        value=1,
                        label="Travail du sol",
                    )

                with gr.Row():
                    density = gr.Number(
                        label="Densité de semis (plants/ha)",
                        value=77500,
                        minimum=60000,
                        maximum=95000,
                    )
                    hybrid_utm = gr.Number(
                        label="Maturité hybride (UTM)",
                        value=2800,
                        minimum=2400,
                        maximum=3200,
                        precision=0,
                        info="Unités Thermiques Maïs (ex: 2800 = maturité moyenne)",
                    )

                gr.Markdown("### Azote du sol")
                with gr.Row():
                    no3n = gr.Number(
                        label="NO3-N au semis (kg N/ha)",
                        value=7.4,
                        minimum=0,
                        maximum=100,
                        info="Azote nitrique mesuré au semis. Laissez vide si non mesuré.",
                    )
                    no3n_imputed = gr.Dropdown(
                        choices=[
                            ("Mesuré", 0),
                            ("Non mesuré (imputé)", 1),
                        ],
                        value=0,
                        label="Statut NO3-N",
                        info="Indiquez si la valeur a été mesurée ou estimée",
                    )

                gr.Markdown("### Dates de culture")
                gr.Markdown(
                    "*Les dates sont utilisées pour aligner les séries temporelles météo et comme caractéristiques du modèle.*"
                )
                with gr.Row():
                    year = gr.Number(
                        label="Année",
                        value=2024,
                        minimum=1990,
                        maximum=2030,
                        precision=0,
                    )
                    seeding_jd = gr.Number(
                        label="Date de semis (jour julien)",
                        value=132,
                        minimum=1,
                        maximum=366,
                        precision=0,
                        info="Jour julien (1-366). Ex: 132 = 12 mai",
                    )
                    harvest_jd = gr.Number(
                        label="Date de récolte (jour julien)",
                        value=280,
                        minimum=1,
                        maximum=366,
                        precision=0,
                        info="Jour julien (1-366). Ex: 280 = 7 octobre",
                    )

                gr.Markdown("### Plage d'azote")
                with gr.Row():
                    n_min = gr.Number(
                        label="Min N (kg/ha)", value=0, minimum=0, maximum=300
                    )
                    n_max = gr.Number(
                        label="Max N (kg/ha)", value=250, minimum=0, maximum=400
                    )
                    n_step = gr.Number(
                        label="Step (kg/ha)", value=10, minimum=1, maximum=50
                    )

                gr.Markdown("### Données météo")
                gr.Markdown(
                    "**Important:** Les fichiers météo doivent être alignés avec les dates de semis et récolte:\n"
                    "- **Météo pré-semis**: Les 30 jours AVANT la date de semis\n"
                    "- **Météo saison de croissance**: Les jours entre semis et récolte (max 200 jours)\n\n"
                    "Voir le répertoire `example_weather/` pour le format attendu."
                )
                with gr.Row():
                    pre_weather = gr.File(
                        label="Météo pré-semis (30 jours)", file_types=[".csv"]
                    )
                    grow_weather = gr.File(
                        label="Météo saison de croissance (150-200 jours)",
                        file_types=[".csv"],
                    )

                gr.Markdown("### Paramètres économiques")
                with gr.Row():
                    n_price = gr.Number(
                        label="Prix N ($/kg)", value=1.5, minimum=0, maximum=5
                    )
                    grain_price = gr.Number(
                        label="Prix grain ($/kg)", value=0.20, minimum=0, maximum=1
                    )

                gr.Markdown("### Options")
                auto_update = gr.Checkbox(
                    label="Mise à jour automatique", value=False
                )

                predict_btn = gr.Button("Lancer la prédiction", variant="primary")

            # Right column - Results
            with gr.Column(scale=2):
                gr.Markdown("### Résultats")
                plot_output = gr.Plot(label="Courbe dose-réponse")
                summary_output = gr.Markdown()

        # List of all inputs
        all_inputs = [
            latitude,
            longitude,
            clay,
            sand,
            ph,
            som,
            prev_crop,
            tillage,
            density,
            hybrid_utm,
            no3n,
            no3n_imputed,
            year,
            seeding_jd,
            harvest_jd,
            n_min,
            n_max,
            n_step,
            pre_weather,
            grow_weather,
            n_price,
            grain_price,
        ]

        # Toggle button interactivity based on auto_update
        def toggle_button(auto_enabled):
            return gr.update(interactive=not auto_enabled)

        auto_update.change(
            fn=toggle_button, inputs=[auto_update], outputs=[predict_btn]
        )

        # Connect prediction on button click
        predict_btn.click(
            fn=predict_yield,
            inputs=all_inputs,
            outputs=[plot_output, summary_output],
        )

        # Connect auto-update on any input change
        def predict_if_auto(*args):
            *input_vals, auto_enabled = args
            if auto_enabled:
                return predict_yield(*input_vals)
            else:
                return None, ""

        for input_component in all_inputs:
            input_component.change(
                fn=predict_if_auto,
                inputs=all_inputs + [auto_update],
                outputs=[plot_output, summary_output],
            )

    return interface


def main():
    parser = argparse.ArgumentParser(description="Maize Yield Prediction Web Interface")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to .keras model file"
    )
    parser.add_argument(
        "--preprocessor", type=str, required=True, help="Path to .pkl preprocessor file"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    # Initialize model
    initialize_model(args.model, args.preprocessor)

    # Create and launch interface
    interface = create_interface()
    interface.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
