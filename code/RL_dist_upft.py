import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

##############################################################################
# 2. Train a Teacher Model (Control)
##############################################################################

def train_teacher_model(df_control):
    """
    Train a complex XGBoost model on the control subset, returning:
      - the trained model,
      - a dictionary of feature importances,
      - the exact list of columns (teacher_col_list) used for training.
    """
    raw_features = [
        "Age", "Income", "DaysSinceLastPurchase",
        "IsHolidaySeason", "PreferredChannel", "LoyaltyScore"
    ]
    X = pd.get_dummies(df_control[raw_features], drop_first=True)
    y = df_control["Purchase"]
    teacher_col_list = list(X.columns)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_val_pred)
    print(f"[Teacher Model] Control Validation AUC: {auc_val:.4f}")
    
    importance_dict = model.get_booster().get_score(importance_type='gain')
    teacher_importance = {col: importance_dict.get(col, 0.0) for col in teacher_col_list}

    return model, teacher_importance, teacher_col_list

##############################################################################
# 3A. Additional Plots for Business: Gains Chart
##############################################################################

def plot_gains(df_wave, model, teacher_col_list, wave_number=0):
    """
    Plots a Gains (Lift) chart for the final wave’s student model predictions.
    This chart shows how many positive responses (purchases) you capture 
    as you move from high to low predicted probability.
    """
    from sklearn.preprocessing import StandardScaler
    # 1) Preprocess wave data
    raw_features = ["Age", "Income", "DaysSinceLastPurchase",
                    "IsHolidaySeason", "PreferredChannel", "LoyaltyScore"]
    X_wave = pd.get_dummies(df_wave[raw_features], drop_first=True)
    X_wave = X_wave.reindex(columns=teacher_col_list, fill_value=0.0)

    y_true = df_wave["Purchase"].values
    
    # Scale similarly (demo approach; in production, re-use the same scaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_wave)

    # 2) Generate predicted probabilities
    preds = model.predict_proba(X_scaled)[:, 1]

    # 3) Sort by predicted probability descending
    sort_idx = np.argsort(-preds)
    y_sorted = y_true[sort_idx]

    # 4) Compute cumulative gains
    gains = np.cumsum(y_sorted) / y_sorted.sum()
    x_vals = np.arange(1, len(y_true) + 1) / len(y_true)

    # 5) Plot Gains chart
    plt.figure(figsize=(8,5))
    plt.plot(x_vals, gains, label="Model")
    plt.plot([0,1],[0,1], 'r--', label="Random")
    plt.title(f"Gains Chart (Wave {wave_number})")
    plt.xlabel("Proportion of Customers (sorted by predicted probability)")
    plt.ylabel("Proportion of Actual Purchases Captured")
    plt.legend()
    plt.grid(True)
    plt.show()

##############################################################################
# 3B. Additional Plots for Business: Decile Analysis
##############################################################################

def decile_analysis(df_wave, model, teacher_col_list, wave_number=0, n_splits=10):
    """
    Splits the wave data into deciles based on predicted probability 
    and shows average predicted probability vs. actual purchase rate in each decile.
    """
    from sklearn.preprocessing import StandardScaler
    # 1) Preprocess wave data
    raw_features = ["Age", "Income", "DaysSinceLastPurchase",
                    "IsHolidaySeason", "PreferredChannel", "LoyaltyScore"]
    X_wave = pd.get_dummies(df_wave[raw_features], drop_first=True)
    X_wave = X_wave.reindex(columns=teacher_col_list, fill_value=0.0)

    y_true = df_wave["Purchase"].values
    
    # Scale similarly (demo approach)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_wave)
    
    # 2) Predicted probabilities
    preds = model.predict_proba(X_scaled)[:, 1]

    # Combine predictions with actuals
    data = pd.DataFrame({"pred": preds, "actual": y_true})
    data.sort_values("pred", ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 3) Create deciles
    data["decile"] = pd.qcut(data.index, n_splits, labels=False)

    # 4) Compute average predicted prob and actual purchase rate per decile
    decile_stats = data.groupby("decile").agg({
        "pred": "mean",
        "actual": "mean"
    }).rename(columns={"pred": "avg_pred", "actual": "actual_rate"})
    
    print(f"\nDecile Analysis for Wave {wave_number} (top decile=0, bottom decile={n_splits-1}):")
    print(decile_stats)

    # 5) Optional: plot decile stats
    decile_stats[["avg_pred","actual_rate"]].plot(
        kind="bar", figsize=(8,5),
        title=f"Wave {wave_number} Decile Analysis: Predicted vs. Actual"
    )
    plt.xlabel(f"Decile (0=top, {n_splits-1}=bottom)")
    plt.ylabel("Rate")
    plt.grid(True)
    plt.show()

##############################################################################
# 4. RL Loop with Uplift-Like Reporting and Plots
##############################################################################

def run_rl_experiment(
    df,
    teacher_model,
    teacher_importance,
    teacher_col_list,
    n_waves=5,
    random_state=42
):
    """
    A naive RL-like loop for multiple 'campaign waves' in the treatment data.
    For each wave, candidate hyperparameters (alpha, beta, gamma) are tried, and the best
    combination (based on reward = -total_loss) is selected.
    Outputs friendly, business-oriented results and plots, plus Gains & Decile analyses.
    """
    np.random.seed(random_state)
    df_treat = df[df["PromotionFlag"] == 1].copy()
    wave_size = len(df_treat) // n_waves

    alpha_candidates = [0.5, 1.0, 2.0]
    beta_candidates = [0.1, 0.5, 1.0]
    gamma_candidates = [0.0, 0.05, 0.1]

    wave_results = []
    # We'll store the best LR model + wave data for the final wave
    final_wave_data = None
    final_wave_best_model = None

    for wave in range(n_waves):
        start_idx = wave * wave_size
        end_idx = len(df_treat) if wave == (n_waves - 1) else (wave + 1) * wave_size
        df_wave = df_treat.iloc[start_idx:end_idx].copy()

        if df_wave.empty:
            print(f"Wave {wave+1}: No data slice available. Skipping.")
            continue

        print(f"\n=== Wave {wave+1} ===")
        best_reward = float("-inf")
        best_params = None
        best_model_info = {}
        best_model = None

        for a in alpha_candidates:
            for b in beta_candidates:
                for g in gamma_candidates:
                    (
                        lr_model,
                        total_loss,
                        alpha_loss,
                        distill_loss,
                        struct_loss,
                        auc_stu
                    ) = train_student_model(
                        df_wave,
                        teacher_model,
                        teacher_importance,
                        teacher_col_list,
                        alpha=a,
                        beta=b,
                        gamma=g
                    )
                    reward = -total_loss
                    if reward > best_reward:
                        best_reward = reward
                        best_params = (a, b, g)
                        best_model_info = {
                            "total_loss": total_loss,
                            "alpha_loss": alpha_loss,
                            "distill_loss": distill_loss,
                            "struct_loss": struct_loss,
                            "auc_student": auc_stu
                        }
                        best_model = lr_model

        print(
            f"Best hyperparams: alpha={best_params[0]}, beta={best_params[1]}, "
            f"gamma={best_params[2]} with reward={best_reward:.4f}"
        )
        print(" -> Breakdown of losses for best model:")
        print(f"    AUC-based loss (alpha_loss) : {best_model_info['alpha_loss']:.4f}")
        print(f"    Distillation loss (beta)     : {best_model_info['distill_loss']:.4f}")
        print(f"    Structure loss (gamma)       : {best_model_info['struct_loss']:.4f}")
        print(f"    Student AUC                  : {best_model_info['auc_student']:.4f}")
        print(f"    TOTAL LOSS                   : {best_model_info['total_loss']:.4f}")

        wave_results.append(
            {
                "wave": wave + 1,
                "alpha": best_params[0],
                "beta": best_params[1],
                "gamma": best_params[2],
                "reward": best_reward,
                "alpha_loss": best_model_info["alpha_loss"],
                "distill_loss": best_model_info["distill_loss"],
                "struct_loss": best_model_info["struct_loss"],
                "auc_student": best_model_info["auc_student"],
                "total_loss": best_model_info["total_loss"],
            }
        )

        # If this is the final wave, store the best wave data + best model
        if wave == n_waves - 1:
            final_wave_data = df_wave
            final_wave_best_model = best_model

    df_summary = pd.DataFrame(wave_results)
    print("\n=========== Summary of RL Campaign Waves ===========")
    print(df_summary)

    # Plot Student AUC vs. Wave
    plt.figure(figsize=(8, 5))
    plt.plot(df_summary["wave"], df_summary["auc_student"], marker="o", linestyle="-")
    plt.title("Student Model AUC Across Campaign Waves")
    plt.xlabel("Campaign Wave")
    plt.ylabel("Student Model AUC")
    plt.ylim(0.5, 0.7)
    plt.grid(True)
    plt.show()

    # Plot Total Loss vs. Wave
    plt.figure(figsize=(8, 5))
    plt.plot(
        df_summary["wave"],
        df_summary["total_loss"],
        marker="o",
        linestyle="-",
        color="red"
    )
    plt.title("Total Loss Across Campaign Waves")
    plt.xlabel("Campaign Wave")
    plt.ylabel("Total Loss")
    plt.ylim(0.15, 0.22)
    plt.grid(True)
    plt.show()

    # If we have final wave data and model, let's do Gains chart + Decile analysis
    if final_wave_data is not None and final_wave_best_model is not None:
        print("\n=== Additional Business-Focused Plots for Final Wave ===")
        # Gains (Lift) Chart
        plot_gains(final_wave_data, final_wave_best_model, teacher_col_list, wave_number=n_waves)
        # Decile Analysis
        decile_analysis(final_wave_data, final_wave_best_model, teacher_col_list, wave_number=n_waves)

    # Business-friendly summary interpretation
    if not df_summary.empty:
        best_entry = df_summary.iloc[-1]
        print("\nFriendly Interpretation:")
        print(
            f"In wave #{int(best_entry['wave'])}, the best approach used hyperparameters "
            f"alpha={best_entry['alpha']}, beta={best_entry['beta']}, gamma={best_entry['gamma']}. "
            f"This student model achieved an AUC of {best_entry['auc_student']:.4f} on the wave's data, "
            "balancing the need to mimic the teacher's predictions while retaining a strong response signal. "
            "This means we've identified a simpler, adaptable model for the treatment group, "
            "which can help us target customers more effectively than a static AB test.\n"
            "Overall, this dynamic approach provides a promising way to refine marketing strategies in real time.\n"
            "Business Suggestion:\n"
            "1. Focus on the top predicted customers in wave #5 for a targeted campaign.\n"
            "2. Consider potential cost savings by limiting promotions to these high-likelihood responders.\n"
            "3. Evaluate real ROI or revenue uplift from these top segments before scaling the approach further."
        )

    return df_summary

##############################################################################
# 5. Main Execution
##############################################################################

if __name__ == "__main__":
   
    # Load data directly from CSV file
    df_all = pd.read_csv(r"C:\backupcgi\final_bak\marketing_data.csv")
    
    # Split into control and treatment subsets
    df_control = df_all[df_all["PromotionFlag"] == 0].copy()
    df_treat = df_all[df_all["PromotionFlag"] == 1].copy()

    # Train teacher model on control data
    teacher_model, teacher_importance, teacher_col_list = train_teacher_model(df_control)

    # Evaluate teacher model performance
    raw_features = [
        "Age", "Income", "DaysSinceLastPurchase",
        "IsHolidaySeason", "PreferredChannel", "LoyaltyScore"
    ]
    X_control = pd.get_dummies(df_control[raw_features], drop_first=True)
    X_control = X_control.reindex(columns=teacher_col_list, fill_value=0.0)
    teacher_preds_control = teacher_model.predict_proba(X_control)[:, 1]
    auc_teacher_control = roc_auc_score(df_control["Purchase"], teacher_preds_control)

    X_treat = pd.get_dummies(df_treat[raw_features], drop_first=True)
    X_treat = X_treat.reindex(columns=teacher_col_list, fill_value=0.0)
    teacher_preds_treat = teacher_model.predict_proba(X_treat)[:, 1]
    auc_teacher_treat = roc_auc_score(df_treat["Purchase"], teacher_preds_treat)

    print(f"\n** Teacher Model AUC on Control = {auc_teacher_control:.4f}")
    print(f"** Teacher Model AUC on Treatment = {auc_teacher_treat:.4f}")
    print(
        "This gives a rough sense of how the data distributions differ between "
        "the control and treatment groups.\n"
    )

    # Run the RL-like loop with uplift-like reporting and generate plots
    run_rl_experiment(
        df_all, teacher_model, teacher_importance, teacher_col_list,
        n_waves=5, random_state=123
    )
