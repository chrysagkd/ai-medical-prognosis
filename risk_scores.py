import numpy as np
import matplotlib.pyplot as plt

# === CHADS-VASC ===
def chads_vasc_score(c, h, a2, d, s2, v, a, sc):
    coef = {'c': 1, 'h': 1, 'a2': 2, 'd': 1, 's2': 2, 'v': 1, 'a': 1, 'sc': 1}
    score = (c * coef['c'] + h * coef['h'] + a2 * coef['a2'] +
             d * coef['d'] + s2 * coef['s2'] + v * coef['v'] +
             a * coef['a'] + sc * coef['sc'])
    return score

# === MELD SCORE ===
def meld_score(creatinine, bilirubin, inr):
    if min(creatinine, bilirubin, inr) <= 0:
        raise ValueError("Οι τιμές πρέπει να είναι μεγαλύτερες από το μηδέν.")
    coef = {'cre': 0.957, 'bil': 0.378, 'inr': 1.12}
    intercept = 0.643
    meld = (coef['cre'] * np.log(creatinine) +
            coef['bil'] * np.log(bilirubin) +
            coef['inr'] * np.log(inr) +
            intercept) * 10
    return meld

# === ASCVD SCORE ===
def ascvd(age, cho, hdl, sbp, smoker, diabetes):
    b = {
        'age': 17.114, 'cho': 0.94, 'hdl': -18.92, 'age_hdl': 4.475,
        'sbp': 27.82, 'age_sbp': -6.087, 'smoker': 0.691, 'diabetes': 0.874
    }

    ln_age = np.log(age)
    ln_cho = np.log(cho)
    ln_hdl = np.log(hdl)
    ln_sbp = np.log(sbp)

    sum_prod = (b['age'] * ln_age +
                b['cho'] * ln_cho +
                b['hdl'] * ln_hdl +
                b['age_hdl'] * ln_age * ln_hdl +
                b['sbp'] * ln_sbp +
                b['age_sbp'] * ln_age * ln_sbp +
                b['smoker'] * smoker +
                b['diabetes'] * diabetes)

    risk = 1 - pow(0.9533, np.exp(sum_prod - 86.61))
    return risk

# === Οπτικοποίηση ===
def plot_scores(scores_dict):
    names = list(scores_dict.keys())
    values = list(scores_dict.values())

    fig, ax = plt.subplots()
    bars = ax.bar(names, values, color=['#4CAF50', '#2196F3', '#FFC107'])
    ax.set_title('Risk Scores Overview')
    ax.set_ylabel('Score')
    ax.set_ylim(0, max(values) + 1)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


# === DEMO ===
if __name__ == "__main__":
    print("📊 Υπολογισμός Risk Scores\n")

    score_chads = chads_vasc_score(c=0, h=1, a2=0, d=0, s2=0, v=1, a=0, sc=1)
    score_meld = meld_score(creatinine=1.0, bilirubin=2.0, inr=1.1)
    score_ascvd = ascvd(age=55, cho=213, hdl=50, sbp=120, smoker=0, diabetes=0)

    print(f"CHADS-VASC Score: {score_chads}")
    print(f"MELD Score: {score_meld:.2f}")
    print(f"ASCVD Risk: {score_ascvd:.2%}")

    plot_scores({
        "CHADS-VASC": score_chads,
        "MELD": score_meld,
        "ASCVD (%)": score_ascvd * 100  # για ομοιομορφία
    })
