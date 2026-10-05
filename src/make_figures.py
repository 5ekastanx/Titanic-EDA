from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "titanic.csv"
FIG_DIR = ROOT / "figures"

PALETTE = {"Погиб": "#C0392B", "Выжил": "#27AE60"}
ORDER = ["Погиб", "Выжил"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Исход"] = df["Survived"].map({0: "Погиб", 1: "Выжил"})
    df["Пол"] = df["Sex"].map({"female": "Женщины", "male": "Мужчины"})
    df["Порт"] = df["Embarked"].map({"Q": "Queenstown", "S": "Southampton", "C": "Cherbourg"})
    return df


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("сохранён", name)


def plot_overview(df):
    mean = df["Survived"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sns.countplot(data=df, x="Pclass", hue="Исход", hue_order=ORDER,
                palette=PALETTE, ax=axes[0, 0])
    sns.barplot(data=df, x="Пол", y="Survived", color="#2E86C1",
                errorbar=None, ax=axes[0, 1])
    sns.histplot(data=df, x="Age", hue="Исход", hue_order=ORDER, palette=PALETTE,
                multiple="stack", bins=30, ax=axes[1, 0])
    sns.boxplot(data=df, x="Исход", y="Fare", order=ORDER, hue="Исход",
                palette=PALETTE, legend=False, ax=axes[1, 1])
    axes[1, 1].set_yscale("log")
    axes[0, 0].set_title("В 3-м классе погибли 3 из 4 пассажиров")
    axes[0, 0].set_xlabel("Класс")
    axes[0, 0].set_ylabel("Количество пассажиров")
    axes[0, 1].set_title("Женщины выживали в 4 раза чаще мужчин")
    axes[0, 1].set_xlabel("Пол")
    axes[0, 1].set_ylabel("Доля выживших")
    axes[1, 0].set_title("Маленькие дети выживали чаще остальных")
    axes[1, 0].set_xlabel("Возраст пассажиров, лет")
    axes[1, 0].set_ylabel("Количество пассажиров")
    axes[1, 1].set_title("Выжившие платили за билет больше\n(медиана 26 £ против 10.5 £)")
    axes[1, 1].set_ylabel("Стоимость билета, £ (лог. шкала)")
    axes[0, 1].axhline(mean, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].text(0.45, mean + 0.02, f"в среднем {mean:.0%}", color="gray", ha="right")
    axes[0, 1].set_ylim(0, 1)

    n_dead_3 = ((df["Pclass"] == 3) & (df["Survived"] == 0)).sum()

    bar = min(
        (p for p in axes[0, 0].patches if p.get_height() == n_dead_3),
        key=lambda p: abs(p.get_x() + p.get_width() / 2 - 2)
    )

    xy = (bar.get_x() + bar.get_width() / 2, bar.get_height())

    axes[0, 0].annotate(
        f"3-й класс: погибли 76% (372 из 491)",
        xy=xy,
        xytext=(0.05, 0.95),
        textcoords="axes fraction",
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color="black")
    )

    for container in axes[0, 1].containers:
        axes[0, 1].bar_label(container, fmt="%.2f")

    fig.suptitle("Титаник: кто выжил", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save(fig, "01_overview.png")


def plot_age(df):
    age_group = pd.cut(df["Age"], bins=[0, 12, 18, 40, 60, 80],
                      labels=["0-12", "12-18", "18-40", "40-60", "60-80"])
    rate = df.pivot_table(index=age_group, columns="Pclass", values="Survived",
                        aggfunc="mean", observed=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(rate, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1, ax=ax)
    ax.set_title("Во 2-м и 3-м классах дети выживали вдвое чаще взрослых")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Возрастная группа, лет")
    save(fig, "02_age.png")


def plot_embarked(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="Порт", hue="Pclass",
                order=["Southampton", "Cherbourg", "Queenstown"], ax=ax)
    ax.set_title("В Cherbourg половина пассажиров из 1-го класса,\nв Queenstown почти все из 3-го")
    ax.legend(title="Класс")
    ax.set_xlabel("Порт посадки")
    ax.set_ylabel("Количество")
    save(fig, "03_embarked.png")


def plot_survival_fare(df):
    fig, ax = plt.subplots(figsize=(7, 5))

    for label in ORDER:
        part = df[df["Исход"] == label]
        ax.scatter(part["Age"], part["Fare"], alpha=0.5, s=20,
                label=label, color=PALETTE[label])

    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Выжившие чаще встречаются среди дорогих билетов")
    ax.set_xlabel("Возраст, лет")
    ax.set_ylabel("Стоимость, £ (лог. шкала)")
    save(fig, "04_survival_fare.png")


def survival_rate_by_class(df):
    mean = df["Survived"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="Pclass", y="Survived", color="#2E86C1", errorbar=None, ax=ax)

    ax.axhline(mean, color="gray", linestyle="--", linewidth=1)
    ax.text(2.45, mean + 0.02, f"в среднем {mean:.0%}", color="gray", ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")

    ax.set_ylim(0, 1)
    ax.set_title("В 1-м классе выживали чаще среднего, в 3-м — реже")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Доля выживших")
    save(fig, "05_survival_rate_by_class.png")


def main():
    sns.set_theme(style="whitegrid")
    FIG_DIR.mkdir(exist_ok=True)
    df = load_data()
    plot_overview(df)
    plot_age(df)
    plot_embarked(df)
    plot_survival_fare(df)
    survival_rate_by_class(df)
if __name__ == "__main__":
    main()