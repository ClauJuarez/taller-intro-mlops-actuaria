import sys
import os
import joblib
import pandas as pd

def main():
    archivo_entrada = sys.argv[1] if len(sys.argv) > 1 else "data/lote_mes_1_inferencia.csv"
    os.makedirs("output", exist_ok=True)
    
    print(f"Cargando modelo y lote: {archivo_entrada}...")
    modelo = joblib.load("models/modelo_retencion.joblib")
    df_lote = pd.read_csv(archivo_entrada)
    
    ids = df_lote["customerID"]
    X = df_lote.drop(columns=["customerID"])
    
    probs = modelo.predict_proba(X)[:, 1]
    
    df_salida = pd.DataFrame({
        "customerID": ids,
        "probabilidad_churn": probs.round(4),
        "alerta_retencion": (probs >= 0.40).astype(int)
    })
    
    df_salida.to_csv("output/calificaciones_lote.csv", index=False)
    
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("# Reporte de Inferencia por Lotes\n\n")
            f.write(f"- **Archivo evaluado:** `{archivo_entrada}`\n")
            f.write(f"- **Clientes calificados:** {len(df_salida):,}\n")
            f.write(f"- **Clientes en alerta (Riesgo >= 40%):** {df_salida['alerta_retencion'].sum():,} ({df_salida['alerta_retencion'].mean():.2%})\n")
            f.write(f"- **Probabilidad media de fuga:** {df_salida['probabilidad_churn'].mean():.4f}\n\n")
            f.write("### Primeros 5 clientes calificados:\n\n")
            f.write(df_salida.head().to_markdown(index=False))
            f.write("\n")

if __name__ == "__main__":
    main()
