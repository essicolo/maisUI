#!/usr/bin/env python3
"""
example_usage.py

Example script demonstrating the maisUI model wrapper API.
Shows how to use the predictor programmatically without the web UI.
"""

from model_wrapper import MaizeYieldPredictor
import pandas as pd
import json


def main():
    """Run example predictions."""

    # 1. Initialize the predictor
    print("=" * 70)
    print("🌽 EXEMPLE D'UTILISATION - maisUI")
    print("=" * 70)
    print("\n1. Chargement du modèle...")

    predictor = MaizeYieldPredictor(
        model_path="../mais-npk/data/models/baseline_model_20250107_164318.keras",
        preprocessor_path="../mais-npk/data/models/baseline_preprocessor_20250107_164318.pkl"
    )
    print("   ✓ Modèle chargé avec succès")

    # 2. Inspect feature schema
    print("\n2. Inspection du schéma des features...")
    schema = predictor.describe_features()

    print(f"   - Version du modèle: {schema['model_version']}")
    print(f"   - Description: {schema['description']}")
    print(f"   - Nombre de features statiques: {len(schema['features']['static'])}")
    print(f"   - Nombre de features météo: {len(schema['features']['time_series'])}")

    print("\n   Features statiques disponibles:")
    for feature in schema['features']['static']:
        print(f"     • {feature['label']} ({feature['name']}): {feature['description']}")

    # 3. Define example field characteristics
    print("\n3. Définition des caractéristiques de la parcelle...")

    features = {
        "latitude": 46.5,           # Québec, région de Québec
        "longitude": -71.5,
        "ph_eau": 6.5,              # Sol légèrement acide
        "som_log_ratio": -3.4,      # ~3% matière organique
        "soil_ilr1": 0.0,           # Texture équilibrée
        "soil_ilr2": 0.0,
        "tillage_ord": 1,           # Labour réduit
        "prev_crop_n": 1,           # Céréale l'année précédente
        "density_norm": 0.0         # Densité standard (77,500 plants/ha)
    }

    print("   Caractéristiques de la parcelle:")
    for key, value in features.items():
        meta = next((f for f in schema['features']['static'] if f['name'] == key), None)
        if meta:
            print(f"     • {meta['label']}: {value} {meta.get('unit', '')}")

    # 4. Predict dose-response curve
    print("\n4. Prédiction de la courbe de réponse...")

    nitrogen_range = list(range(0, 301, 25))  # 0 à 300 kg N/ha par pas de 25
    print(f"   Plage d'azote testée: {min(nitrogen_range)} - {max(nitrogen_range)} kg N/ha")

    dose_response = predictor.predict_response_curve(
        features=features,
        nitrogen_range=nitrogen_range,
        weather_data=None  # Utilise des valeurs par défaut
    )

    print(f"   ✓ {len(dose_response)} prédictions générées")

    # Display results
    print("\n   Résultats (extrait):")
    print(dose_response.to_string(index=False))

    # 5. Find optimal nitrogen dose
    print("\n5. Recherche de la dose optimale...")

    # Economic parameters
    n_price = 1.5      # $/kg N
    grain_price = 0.20  # $/kg maïs grain

    print(f"   Paramètres économiques:")
    print(f"     • Prix de l'azote: {n_price} $/kg N")
    print(f"     • Prix du grain: {grain_price} $/kg")

    optimal = predictor.find_optimal_nitrogen(
        dose_response=dose_response,
        n_price=n_price,
        grain_price=grain_price
    )

    print("\n   ⭐ DOSE OPTIMALE RECOMMANDÉE:")
    print(f"     • Dose d'azote: {optimal['optimal_n_kg_ha']:.0f} kg N/ha")
    print(f"     • Rendement prédit: {optimal['predicted_yield_t_ha']:.2f} t/ha")
    print(f"     • Revenu: {optimal['revenue_per_ha']:.0f} $/ha")
    print(f"     • Coût fertilisation: {optimal['cost_per_ha']:.0f} $/ha")
    print(f"     • Revenu net: {optimal['net_revenue_per_ha']:.0f} $/ha")

    # 6. Save results
    print("\n6. Sauvegarde des résultats...")

    # Save dose-response curve
    dose_response.to_csv("dose_response_example.csv", index=False)
    print("   ✓ Courbe sauvegardée: dose_response_example.csv")

    # Save optimal recommendation
    with open("optimal_recommendation.json", "w", encoding="utf-8") as f:
        json.dump({
            "features": features,
            "optimal": optimal,
            "economic_params": {
                "n_price": n_price,
                "grain_price": grain_price
            }
        }, f, indent=2, ensure_ascii=False)
    print("   ✓ Recommandation sauvegardée: optimal_recommendation.json")

    # 7. Example with weather data
    print("\n7. Exemple avec données météo (simulation)...")

    # Create dummy weather data
    # In practice, this would come from ERA5, weather station, or CSV upload
    n_weather_features = len(predictor.preprocessor.weather_cols)

    # Pre-seedling weather (30 days)
    pre_weather_df = pd.DataFrame(
        data=0.5,  # Placeholder values (would be real weather data)
        index=range(predictor.preprocessor.pre_seedling_days),
        columns=predictor.preprocessor.weather_cols
    )

    # Growing season weather (200 days)
    growing_weather_df = pd.DataFrame(
        data=0.5,
        index=range(predictor.preprocessor.max_growing_days),
        columns=predictor.preprocessor.weather_cols
    )

    weather_data = {
        'pre_seedling': pre_weather_df,
        'growing': growing_weather_df
    }

    print(f"   Données météo:")
    print(f"     • Pré-semis: {len(pre_weather_df)} jours x {len(predictor.preprocessor.weather_cols)} variables")
    print(f"     • Saison: {len(growing_weather_df)} jours x {len(predictor.preprocessor.weather_cols)} variables")

    # Predict with weather data
    dose_response_weather = predictor.predict_response_curve(
        features=features,
        nitrogen_range=nitrogen_range,
        weather_data=weather_data
    )

    print(f"   ✓ Prédictions avec météo générées")

    optimal_weather = predictor.find_optimal_nitrogen(
        dose_response=dose_response_weather,
        n_price=n_price,
        grain_price=grain_price
    )

    print(f"   Dose optimale avec météo: {optimal_weather['optimal_n_kg_ha']:.0f} kg N/ha")
    print(f"   Rendement prédit: {optimal_weather['predicted_yield_t_ha']:.2f} t/ha")

    # Compare with and without weather
    print("\n8. Comparaison avec/sans données météo:")
    print(f"   Sans météo: {optimal['predicted_yield_t_ha']:.2f} t/ha à {optimal['optimal_n_kg_ha']:.0f} kg N/ha")
    print(f"   Avec météo: {optimal_weather['predicted_yield_t_ha']:.2f} t/ha à {optimal_weather['optimal_n_kg_ha']:.0f} kg N/ha")

    diff_yield = optimal_weather['predicted_yield_t_ha'] - optimal['predicted_yield_t_ha']
    diff_n = optimal_weather['optimal_n_kg_ha'] - optimal['optimal_n_kg_ha']

    print(f"   Différence rendement: {diff_yield:+.2f} t/ha")
    print(f"   Différence dose: {diff_n:+.0f} kg N/ha")

    print("\n" + "=" * 70)
    print("✅ EXEMPLE TERMINÉ")
    print("=" * 70)
    print("\nFichiers générés:")
    print("  • dose_response_example.csv")
    print("  • optimal_recommendation.json")
    print("\nPour l'interface web complète, lancez:")
    print("  python app.py")


if __name__ == "__main__":
    main()
