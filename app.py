import math

import pandas as pd
import streamlit as st

BASE = "https://www.football-data.co.uk"
SEASONS = ["2526", "2627"]  # последните 2 сезона за главните първенства

MAIN = {
    "E0": "Англия - Premier League", "E1": "Англия - Championship",
    "SC0": "Шотландия - Premiership",
    "D1": "Германия - Бундеслига", "D2": "Германия - 2. Бундеслига",
    "I1": "Италия - Серия А", "I2": "Италия - Серия Б",
    "SP1": "Испания - Ла Лига", "SP2": "Испания - Сегунда",
    "F1": "Франция - Лига 1", "F2": "Франция - Лига 2",
    "N1": "Нидерландия - Ередивизи", "B1": "Белгия - Про Лига",
    "P1": "Португалия - Примейра", "T1": "Турция - Суперлига",
    "G1": "Гърция - Суперлига",
}
EXTRA = {
    "ARG": "Аржентина", "AUT": "Австрия", "BRA": "Бразилия", "CHN": "Китай",
    "DNK": "Дания", "FIN": "Финландия", "IRL": "Ирландия", "JPN": "Япония",
    "MEX": "Мексико", "NOR": "Норвегия", "POL": "Полша", "ROU": "Румъния",
    "RUS": "Русия", "SWE": "Швеция", "SWZ": "Швейцария", "USA": "САЩ",
}
ALL = {**MAIN, **EXTRA}


# ---------------- ДАННИ ----------------
@st.cache_data(ttl=6 * 3600, show_spinner="Свалям резултатите...")
def load_league(code):
    """Сваля данните за първенство (пазят се 6 часа, за да е бързо)."""
    if code in MAIN:
        frames = []
        for s in SEASONS:
            try:
                df = pd.read_csv(f"{BASE}/mmz4281/{s}/{code}.csv", encoding="latin-1")
                df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
                df["Season"] = s
                frames.append(df)
            except Exception:
                pass
        if not frames:
            return pd.DataFrame(columns=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"])
        return pd.concat(frames, ignore_index=True)

    df = pd.read_csv(f"{BASE}/new/{code}.csv", encoding="latin-1")
    df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
    df["Season"] = df["Season"].astype(str)
    last_two = sorted(df["Season"].unique())[-2:]
    return df[df["Season"].isin(last_two)][["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"]].dropna()


@st.cache_data(ttl=3 * 3600, show_spinner="Свалям програмата...")
def load_fixtures():
    return pd.read_csv(f"{BASE}/fixtures.csv", encoding="latin-1")


# ---------------- МОДЕЛ ----------------
def poisson(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


def team_strengths(df):
    home_avg, away_avg = df["FTHG"].mean(), df["FTAG"].mean()
    h = df.groupby("HomeTeam")[["FTHG", "FTAG"]].mean()
    a = df.groupby("AwayTeam")[["FTHG", "FTAG"]].mean()
    s = pd.DataFrame({
        "Атака дома": h["FTHG"] / home_avg,
        "Защита дома": h["FTAG"] / away_avg,
        "Атака навън": a["FTAG"] / away_avg,
        "Защита навън": a["FTHG"] / home_avg,
    })
    return s, home_avg, away_avg


def calc(df, home, away):
    s, home_avg, away_avg = team_strengths(df)
    if home not in s.index or away not in s.index:
        return None
    exp_h = s.loc[home, "Атака дома"] * s.loc[away, "Защита навън"] * home_avg
    exp_a = s.loc[away, "Атака навън"] * s.loc[home, "Защита дома"] * away_avg
    p1 = px = p2 = 0
    scores = []
    for i in range(8):
        for j in range(8):
            p = poisson(i, exp_h) * poisson(j, exp_a)
            scores.append((f"{i}:{j}", p))
            if i > j:
                p1 += p
            elif i == j:
                px += p
            else:
                p2 += p
    t = p1 + px + p2
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    return {
        "exp_h": exp_h, "exp_a": exp_a,
        "1": p1 / t, "X": px / t, "2": p2 / t,
        "top": [(r, p / t) for r, p in top],
    }


# ---------------- САЙТ ----------------
st.set_page_config(page_title="Футболни прогнози", page_icon="⚽", layout="wide")
st.title("⚽ Футболни прогнози")
st.caption("Поасонов модел върху резултатите от последните два сезона. Само за забавление, не за залози.")

tab_match, tab_week, tab_table = st.tabs(["Прогноза за мач", "Мачове тази седмица", "Сила на отборите"])

# --- Прогноза за един мач ---
with tab_match:
    code = st.selectbox("Първенство", list(ALL), format_func=lambda c: ALL[c])
    try:
        data = load_league(code)
    except Exception:
        data = pd.DataFrame()
    if data.empty:
        st.error("Не успях да сваля данни за това първенство.")
    else:
        team_list = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
        c1, c2 = st.columns(2)
        home = c1.selectbox("Домакин", team_list, index=0)
        away = c2.selectbox("Гост", team_list, index=min(1, len(team_list) - 1))

        if home == away:
            st.warning("Избери два различни отбора.")
        else:
            r = calc(data, home, away)
            if r is None:
                st.warning("Няма достатъчно мачове за някой от отборите.")
            else:
                st.subheader(f"{home} - {away}")
                st.write(f"Очаквани голове: **{r['exp_h']:.2f} - {r['exp_a']:.2f}**")
                m1, m2, m3 = st.columns(3)
                m1.metric(f"1 · {home}", f"{r['1']:.0%}")
                m2.metric("X · равен", f"{r['X']:.0%}")
                m3.metric(f"2 · {away}", f"{r['2']:.0%}")

                st.write("**Най-вероятни резултати**")
                top = pd.DataFrame(r["top"], columns=["Резултат", "Вероятност"])
                top["Вероятност"] = top["Вероятност"].map(lambda p: f"{p:.0%}")
                st.dataframe(top, hide_index=True, use_container_width=False)

# --- Всички предстоящи мачове ---
with tab_week:
    st.write("Прогнози за предстоящите мачове в главните европейски първенства.")
    if st.button("Покажи прогнозите"):
        try:
            fx = load_fixtures()
        except Exception:
            fx = pd.DataFrame()
        rows = []
        for _, m in fx.iterrows():
            c = m.get("Div")
            if c not in MAIN:
                continue
            try:
                r = calc(load_league(c), m["HomeTeam"], m["AwayTeam"])
            except Exception:
                r = None
            if r is None:
                continue
            best = "1" if r["1"] >= max(r["X"], r["2"]) else ("X" if r["X"] >= r["2"] else "2")
            rows.append({
                "Дата": m["Date"], "Първенство": MAIN[c],
                "Домакин": m["HomeTeam"], "Гост": m["AwayTeam"],
                "1": f"{r['1']:.0%}", "X": f"{r['X']:.0%}", "2": f"{r['2']:.0%}",
                "Фаворит": best, "Резултат": r["top"][0][0],
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("В момента няма предстоящи мачове в програмата.")

# --- Сила на отборите ---
with tab_table:
    code_t = st.selectbox("Първенство ", list(ALL), format_func=lambda c: ALL[c], key="table_league")
    try:
        d = load_league(code_t)
    except Exception:
        d = pd.DataFrame()
    if d.empty:
        st.error("Не успях да сваля данни за това първенство.")
    else:
        s, _, _ = team_strengths(d)
        st.caption("Атака над 1.0 = вкарва повече от средното. Защита под 1.0 = допуска по-малко от средното.")
        st.dataframe(s.sort_values("Атака дома", ascending=False).round(2), use_container_width=True)
