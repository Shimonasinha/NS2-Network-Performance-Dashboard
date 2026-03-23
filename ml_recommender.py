# ml_recommender.py - Restored version (gives 80-81% accuracy)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, pickle, warnings, json
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import classification_report, confusion_matrix, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except: HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except: HAS_SHAP = False

FEATURES = [
    "throughput_mbps","mean_rtt_ms","p90_rtt_ms","p99_rtt_ms",
    "std_rtt_ms","loss_rate","retx_rate",
    "flow_count","network_load","app_type_enc"
]

FEAT_NAMES = [
    "throughput","mean_rtt","p90_rtt","p99_rtt","std_rtt",
    "loss_rate","retx_rate","flow_count","net_load","app_type"
]

COLORS = {
    "reno":"#e74c3c","tahoe":"#3498db","vegas":"#2ecc71",
    "cubic":"#f39c12","bbr":"#9b59b6","dctcp":"#1abc9c"
}

def load_data():
    if not os.path.exists("tcp_dataset.csv"):
        print(" Run generate_dataset.py first!"); exit(1)
    df               = pd.read_csv("tcp_dataset.csv")
    le_app           = LabelEncoder()
    le_label         = LabelEncoder()
    df["app_type_enc"]      = le_app.fit_transform(df["app_type"])
    df["best_protocol_enc"] = le_label.fit_transform(df["best_protocol"])
    print(f"  Dataset loaded: {len(df)} samples")
    print(f"  Classes: {list(le_label.classes_)}")
    print(f"  Features: {len(FEATURES)}")
    return df, le_app, le_label

def train_models(X_train, y_train):
    models = {}
    print("\n→ Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                 min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    print("   Done")

    print("→ Training Logistic Regression...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr       = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y_train)
    models["Logistic Regression"] = (lr, scaler)
    print("   Done")

    if HAS_XGB:
        print("→ Training XGBoost...")
        xgb = XGBClassifier(n_estimators=100, max_depth=4,
                             learning_rate=0.1, random_state=42,
                             eval_metric='mlogloss', verbosity=0,
                             subsample=0.8, colsample_bytree=0.8)
        xgb.fit(X_train, y_train)
        models["XGBoost"] = xgb
        print("   Done")

    return models

def evaluate(models, X_test, y_test, le_label):
    results = {}
    print("\n"+"="*55)
    print("        MODEL EVALUATION RESULTS")
    print("="*55)
    for name, model in models.items():
        if isinstance(model, tuple):
            lr, scaler = model
            X_s = scaler.transform(X_test)
            preds = lr.predict(X_s)
        else:
            preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"accuracy":acc, "preds":preds}
        print(f"\n  📊 {name}")
        print(f"     Test Accuracy : {acc*100:.2f}%")
        print(f"\n{classification_report(y_test, preds, target_names=le_label.classes_)}")
    return results

def plot_results(models, results, X_test, y_test, df, le_label):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("ML-Based TCP Variant Recommendation Engine",
                 fontsize=14, fontweight='bold')

    # 1. Accuracy comparison
    ax1 = fig.add_subplot(3, 3, 1)
    names = list(results.keys())
    accs  = [results[n]["accuracy"]*100 for n in names]
    bars  = ax1.bar(names, accs,
                    color=["#2980b9","#27ae60","#e74c3c"][:len(names)])
    ax1.set_title("Model Accuracy Comparison")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 115)
    ax1.grid(True, alpha=0.3, axis='y')
    for b, a in zip(bars, accs):
        ax1.text(b.get_x()+b.get_width()/2, a+1,
                 f"{a:.1f}%", ha='center', fontsize=9)
    plt.setp(ax1.get_xticklabels(), rotation=10, fontsize=8)

    # 2. Confusion matrix
    ax2 = fig.add_subplot(3, 3, 2)
    preds = results["Random Forest"]["preds"]
    cm    = confusion_matrix(y_test, preds)
    im    = ax2.imshow(cm, cmap='Blues')
    cls   = le_label.classes_
    ax2.set_xticks(range(len(cls))); ax2.set_yticks(range(len(cls)))
    ax2.set_xticklabels(cls, rotation=45, fontsize=7)
    ax2.set_yticklabels(cls, fontsize=7)
    ax2.set_title("Confusion Matrix (Random Forest)")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    for i in range(len(cls)):
        for j in range(len(cls)):
            ax2.text(j, i, cm[i,j], ha='center', va='center',
                     color='white' if cm[i,j]>cm.max()/2 else 'black',
                     fontweight='bold', fontsize=8)
    plt.colorbar(im, ax=ax2)

    # 3. Feature importance
    ax3 = fig.add_subplot(3, 3, 3)
    rf          = models["Random Forest"]
    importances = rf.feature_importances_
    idx         = np.argsort(importances)[::-1]
    ax3.barh([FEAT_NAMES[i] for i in idx],
             importances[idx], color="#8e44ad", alpha=0.8)
    ax3.set_title("Feature Importance (Random Forest)")
    ax3.set_xlabel("Importance Score")
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Label distribution
    ax4 = fig.add_subplot(3, 3, 4)
    counts = df["best_protocol"].value_counts()
    ax4.bar(counts.index, counts.values,
            color=[COLORS.get(p,"#95a5a6") for p in counts.index])
    ax4.set_title("Label Distribution")
    ax4.set_ylabel("Count")
    plt.setp(ax4.get_xticklabels(), rotation=20, fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Throughput vs RTT
    ax5 = fig.add_subplot(3, 3, 5)
    for proto, grp in df.groupby("best_protocol"):
        ax5.scatter(grp["throughput_mbps"], grp["mean_rtt_ms"],
                    alpha=0.2, s=12,
                    color=COLORS.get(proto,"#95a5a6"), label=proto)
    ax5.set_title("Throughput vs Latency")
    ax5.set_xlabel("Throughput (Mbps)")
    ax5.set_ylabel("Mean RTT (ms)")
    ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

    # 6. Avg loss rate per protocol
    ax6 = fig.add_subplot(3, 3, 6)
    proto_list = sorted(df["best_protocol"].unique())
    loss_means = [df[df["best_protocol"]==p]["loss_rate"].mean() for p in proto_list]
    loss_stds  = [df[df["best_protocol"]==p]["loss_rate"].std()  for p in proto_list]
    ax6.bar(proto_list, loss_means, yerr=loss_stds,
            color=[COLORS.get(p,"#95a5a6") for p in proto_list],
            capsize=5, alpha=0.8)
    ax6.set_title("Avg Loss Rate per Protocol")
    ax6.set_ylabel("Loss Rate")
    plt.setp(ax6.get_xticklabels(), rotation=20, fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Loss rate boxplot
    ax7 = fig.add_subplot(3, 3, 7)
    data_by_proto = [df[df["best_protocol"]==p]["loss_rate"].values
                     for p in proto_list]
    bp = ax7.boxplot(data_by_proto, patch_artist=True, labels=proto_list)
    for patch, p in zip(bp['boxes'], proto_list):
        patch.set_facecolor(COLORS.get(p,"#95a5a6"))
        patch.set_alpha(0.7)
    ax7.set_title("Loss Rate Distribution")
    ax7.set_ylabel("Loss Rate")
    plt.setp(ax7.get_xticklabels(), rotation=20, fontsize=8)
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Data source
    ax8 = fig.add_subplot(3, 3, 8)
    src    = df["data_source"].value_counts()
    colors = {"ns2":"#3498db","iperf3":"#e74c3c","synthetic":"#2ecc71"}
    ax8.bar(src.index, src.values,
            color=[colors.get(s,"#95a5a6") for s in src.index])
    ax8.set_title("Data Source Breakdown")
    ax8.set_ylabel("Samples")
    for i,(idx2,v) in enumerate(src.items()):
        ax8.text(i, v+5, str(v), ha='center', fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Per-class accuracy
    ax9 = fig.add_subplot(3, 3, 9)
    preds = results["XGBoost"]["preds"] if "XGBoost" in results \
            else results["Random Forest"]["preds"]
    per_class = []
    for i, c in enumerate(cls):
        mask = y_test == i
        acc2 = accuracy_score(y_test[mask], preds[mask])*100 \
               if mask.sum() > 0 else 0
        per_class.append(acc2)
    ax9.bar(cls, per_class,
            color=[COLORS.get(c,"#95a5a6") for c in cls])
    ax9.set_title("Per-Class Accuracy (XGBoost)")
    ax9.set_ylabel("Accuracy (%)")
    ax9.set_ylim(0, 115)
    plt.setp(ax9.get_xticklabels(), rotation=20, fontsize=8)
    ax9.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(per_class):
        ax9.text(i, v+1, f"{v:.0f}%", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig("ml_results.png", dpi=150, bbox_inches='tight')
    print("\n   Plots saved → ml_results.png")

def demo_predict(model, le_label, le_app):
    print("\n"+"="*55)
    print("  DEMO REAL-TIME PREDICTION")
    print("="*55)
    cases = [
        ("High BW Streaming",   [9.2, 28,  35,  42,  4,   0.008, 0.006, 30, 0.5, "streaming"]),
        ("High Loss Network",   [4.5, 180, 210, 230, 30,  0.09,  0.07,  20, 0.9, "io"]),
        ("Datacenter App",      [9.8, 0.7, 0.9, 1.1, 0.08,0.0008,0.0006,45, 0.6, "sort"]),
        ("Moderate Conditions", [6.5, 148, 175, 190, 19,  0.04,  0.032, 25, 0.5, "io"]),
    ]
    for name, vals in cases:
        app_enc = le_app.transform([vals[-1]])[0]
        X = np.array([vals[:-1] + [app_enc]], dtype=float)
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        rec   = le_label.inverse_transform([pred])[0]
        print(f"\n   {name}")
        print(f"   Recommended: {rec.upper()}")
        for cls2, p in sorted(zip(le_label.classes_, proba),
                               key=lambda x: x[1], reverse=True)[:3]:
            bar = "█" * int(p*25)
            print(f"     {cls2:8s} {bar} {p*100:.1f}%")

def main():
    print("="*55)
    print("  ML-BASED TCP VARIANT RECOMMENDATION ENGINE")
    print("  6 Variants: Reno/Tahoe/Vegas/Cubic/BBR/DCTCP")
    print("="*55)

    df, le_app, le_label = load_data()
    X = df[FEATURES].values
    y = df["best_protocol_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n  Training: {len(X_train)} | Testing: {len(X_test)}")

    models  = train_models(X_train, y_train)
    results = evaluate(models, X_test, y_test, le_label)

    # Save best model
    best_name  = max(results, key=lambda n: results[n]["accuracy"])
    best_model = models[best_name]
    if isinstance(best_model, tuple):
        best_model = best_model[0]

    with open("tcp_model.pkl","wb") as f:
        pickle.dump({"model":best_model,"le_label":le_label,
                     "le_app":le_app,"features":FEATURES}, f)
    print(f"\n   Best model saved: {best_name} → tcp_model.pkl")

    plot_results(models, results, X_test, y_test, df, le_label)

    # ── Save results to JSON for dashboard ──────────────────────
    json_output = {
        "models": {},
        "dataset": {
            "total_samples": int(len(df)),
            "variants":      int(df["best_protocol"].nunique()),
            "label_counts":  {k:int(v) for k,v in df["best_protocol"].value_counts().items()},
            "source_counts": {k:int(v) for k,v in df["data_source"].value_counts().items()},
        }
    }
    for name, res in results.items():
        preds = res["preds"]
        acc   = res["accuracy"]
        per_class = {}
        for i, cls in enumerate(le_label.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                per_class[cls] = round(float(accuracy_score(y_test[mask], preds[mask])*100),2)
            else:
                per_class[cls] = 0.0
        json_output["models"][name] = {
            "accuracy":  round(float(acc)*100, 2),
            "per_class": per_class,
        }
    with open("ml_results.json","w") as f:
        json.dump(json_output, f, indent=2)
    print("\n   Results saved → ml_results.json")

    demo_predict(best_model, le_label, le_app)

    print("\n"+"="*55)
    print("   ML Training Complete!")
    print("  → tcp_model.pkl   (trained model)")
    print("  → ml_results.png  (all plots)")
    print("="*55)

if __name__ == "__main__":
    main()


# ── Save results to JSON for dashboard ────────────────────────────
def save_results_json(models, results, le_label, df):
    import json
    from sklearn.metrics import confusion_matrix

    output = {
        "models": {},
        "dataset": {
            "total_samples":    int(len(df)),
            "variants":         int(df["best_protocol"].nunique()),
            "label_counts":     {k: int(v) for k,v in df["best_protocol"].value_counts().items()},
            "source_counts":    {k: int(v) for k,v in df["data_source"].value_counts().items()},
        }
    }

    for name, model in models.items():
        preds = results[name]["preds"]
        acc   = results[name]["accuracy"]

        # Per class accuracy
        per_class = {}
        for i, cls in enumerate(le_label.classes_):
            mask = results.get("y_test", np.array([])) == i
            if hasattr(mask, '__len__') and mask.sum() > 0:
                per_class[cls] = round(float(
                    accuracy_score(results.get("y_test", preds)[mask], preds[mask]) * 100
                ), 2)
            else:
                per_class[cls] = 0.0

        output["models"][name] = {
            "accuracy":  round(float(acc) * 100, 2),
            "per_class": per_class,
        }

    with open("ml_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("   Results saved → ml_results.json")