"""Optuna로 LightGBM 하이퍼파라미터 + 최근 월 가중 반감기를 시간 순서 백테스트에서 탐색.

- 목적함수: 폴드 A/B/C 평균 로그손실(매끄러워 오분류율보다 잡음이 작다). 오분류율은 기록만 한다.
- 반복수: 학습률 0.05로 MAX_ROUNDS까지 학습하고 100회 간격으로 평가해, 세 폴드 평균이 최소인 반복수를 고른다.
- 폴드 A가 끝나면 중간값 가지치기(MedianPruner).
- SQLite에 저장하므로 같은 명령으로 다시 실행하면 이어서 탐색한다.

사용법: python 06_optuna_lgb.py [시행수]
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score, log_loss
from kcb_common import load, feature_cols, FOLDS, weighted_error, ROOT

warnings.filterwarnings("ignore")
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
LR, MAX_ROUNDS, STEP = 0.05, 1500, 100
CHECKPOINTS = list(range(STEP, MAX_ROUNDS + 1, STEP))
THRS = [0.4, 0.425, 0.45, 0.475, 0.5]
STORAGE = "sqlite:///" + os.path.join(ROOT, "results", "06_optuna_lgb.db").replace("\\", "/")

df = load()
feats = feature_cols(df)
FOLD_DATA = []
for f in FOLDS:
    tr, te = df[df.LNMON <= f["train_end"]], df[df.LNMON > f["train_end"]]
    age = ((f["train_end"] // 100 - tr.LNMON // 100) * 12 + f["train_end"] % 100 - tr.LNMON % 100).values
    FOLD_DATA.append(dict(name=f["name"], Xtr=tr[feats], ytr=tr.TARGET.values, age=age,
                          Xte=te[feats], yte=te.TARGET.values, lnmon=te.LNMON.values))
del df


def objective(trial):
    params = dict(
        objective="binary", learning_rate=LR, verbose=-1, num_threads=8, seed=0, bagging_freq=1,
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 50, 1500, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.15, 0.8),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-2, 100, log=True),
        min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    )
    half_life = trial.suggest_categorical("half_life", [0, 2, 3, 4, 6, 9, 12])  # 0 = 가중 없음
    curves, probs = [], []
    t0 = time.time()
    for i, d in enumerate(FOLD_DATA):
        w = 0.5 ** (d["age"] / half_life) if half_life else None
        m = lgb.train(params, lgb.Dataset(d["Xtr"], d["ytr"], weight=w), MAX_ROUNDS)
        ps = [m.predict(d["Xte"], num_iteration=k) for k in CHECKPOINTS]
        curves.append([log_loss(d["yte"], p) for p in ps])
        probs.append(ps)
        trial.report(min(curves[-1]), step=i)  # 폴드별 최선값으로 가지치기 판단
        if trial.should_prune():
            raise optuna.TrialPruned()
    mean_curve = np.mean(curves, axis=0)
    k = int(np.argmin(mean_curve))
    trial.set_user_attr("rounds", CHECKPOINTS[k])
    trial.set_user_attr("sec", round(time.time() - t0))
    trial.set_user_attr("auc", float(np.mean([roc_auc_score(d["yte"], probs[i][k]) for i, d in enumerate(FOLD_DATA)])))
    for t in THRS:
        errs = [weighted_error(d["lnmon"], d["yte"], (probs[i][k] > t).astype(int))[0] for i, d in enumerate(FOLD_DATA)]
        trial.set_user_attr("err@%.3f" % t, float(np.mean(errs)))
    return float(mean_curve[k])


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="lgb_v1", storage=STORAGE, load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    if len(study.trials) == 0:  # 지금까지의 기준 설정을 첫 시행으로 넣어 비교 기준으로 삼는다
        study.enqueue_trial(dict(num_leaves=31, min_child_samples=200, feature_fraction=0.5, bagging_fraction=0.8,
                                 lambda_l1=1e-3, lambda_l2=10, min_gain_to_split=0.0, half_life=3))

    def log(study, trial):
        a = trial.user_attrs
        print("#%d %s logloss=%s rounds=%s auc=%s err@0.45=%s (%ss) %s" % (
            trial.number, trial.state.name, None if trial.value is None else round(trial.value, 5),
            a.get("rounds"), round(a["auc"], 4) if "auc" in a else None,
            round(a["err@0.450"], 4) if "err@0.450" in a else None, a.get("sec"), trial.params), flush=True)

    done = sum(t.state.is_finished() for t in study.trials)
    study.optimize(objective, n_trials=max(0, N_TRIALS - done), callbacks=[log])

    res = study.trials_dataframe()
    res = res[res.state == "COMPLETE"].sort_values("value")
    res.to_csv(os.path.join(ROOT, "results", "06_optuna_lgb_trials.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["number", "value", "user_attrs_rounds", "user_attrs_auc", "user_attrs_err@0.425", "user_attrs_err@0.450",
            "user_attrs_err@0.500"] + [c for c in res.columns if c.startswith("params_")]
    print("\n=== 상위 10개 시행 ===")
    print(res[cols].head(10).round(4).to_string(index=False))
