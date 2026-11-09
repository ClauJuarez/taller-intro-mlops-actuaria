import sys
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score

BASELINE_AUC = 0.8484
BASELINE_BRIER = 0.1624
AUC_MIN_TOLERABLE = 0.7200
BRIER_MAX_TOLERABLE = 0.2200

def main():
    archivo_predicciones = "output/calificaciones_lote.csv"
    archivo_ground_truth = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth_mes_1.csv"
    
    if not os.path.exists(archivo_predicciones):
        raise FileNotFoundError("Primero debe ejecutarse el script de inferencia.")
        
    df_pred = pd.read_csv(archivo_predicciones)
    df_real = pd.read_csv(archivo_ground_truth)
    df_eval = pd.merge(df_pred, df_real, on="customerID")
    
    y_true = df_eval["Churn"]
    y_prob = df_eval["probabilidad_churn"]
    y_pred = df_eval["alerta_retencion"]
    
    auc_obs = roc_auc_score(y_true, y_prob)
    brier_obs = brier_score_loss(y_true, y_prob)
    recall_obs = recall_score(y_true, y_pred)
    tasa_real = y_true.mean()
    
    alerta_roja = (auc_obs < AUC_MIN_TOLERABLE) or (brier_obs > BRIER_MAX_TOLERABLE)
    
    reporte = [
        {"Métrica": "ROC-AUC (Discriminación)", "Baseline": f"{BASELINE_AUC:.4f}", "Mes Observado": f"{auc_obs:.4f}", "Estado": "🚨 Alerta Crítica" if auc_obs < AUC_MIN_TOLERABLE else "✅ Estable"},
        {"Métrica": "Brier Score (Calibración)", "Baseline": f"{BASELINE_BRIER:.4f}", "Mes Observado": f"{brier_obs:.4f}", "Estado": "🚨 Descalibrado" if brier_obs > BRIER_MAX_TOLERABLE else "✅ Estable"},
        {"Métrica": "Recall (Detección)", "Baseline": "~0.80", "Mes Observado": f"{recall_obs:.4f}", "Estado": "Informativo"},
        {"Métrica": "Tasa Real de Churn", "Baseline": "26.54%", "Mes Observado": f"{tasa_real:.2%}", "Estado": "Informativo"}
    ]
    df_reporte = pd.DataFrame(reporte)
    
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("# 📉 Monitoreo de Desempeño Real (*Ground Truth Lag*)\n\n")
            f.write(f"- **Archivo evaluado:** `{archivo_ground_truth}`\n")
            f.write(f"- **Pólizas analizadas:** {len(df_eval):,}\n\n")
            f.write(df_reporte.to_markdown(index=False))
            f.write("\n\n")
            
            if alerta_roja:
                f.write("### 🚨 ACCIÓN REQUERIDA: REENTRENAMIENTO INMEDIATO\n")
                f.write("> **Diagnóstico Actuarial:** La relación estadística entre las variables y el abandono se ha roto. El riesgo está subestimado; se debe recalibrar el modelo con datos recientes.\n")
            else:
                f.write("### ✅ MODELO SALUDABLE\n")
                f.write("> El desempeño se encuentra dentro de las tolerancias fijadas por el comité de riesgo actuarial.\n")

if __name__ == "__main__":
    main()
