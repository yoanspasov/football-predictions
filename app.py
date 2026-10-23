import math

import pandas as pd
import streamlit as st

BASE = "https://www.football-data.co.uk"
SEASONS = ["2425", "2526", "2627"]  # последните 3 сезона за главните първенства
SHRINK = 2  # колко "средни мача" добавяме към всеки отбор, за да не са крайни оценките при малко данни

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

# Тежест на формата: след колко дни един мач тежи наполовина
FORM_OPTIONS = {
    "Без (всички мачове тежат еднакво)": None,
    "Слаба (половин тежест след 1 година)": 365,
    "Нормална (половин тежест след 6 месеца)": 180,
    "Силна (половин тежест след 3 месеца)": 90,
    "Много силна (половин тежест след 6 седмици)": 45,
}
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "OddsH", "OddsD", "OddsA"]


# ---------------- ДАННИ ----------------
def clean_columns(df):
    """Маха невидимия знак (BOM) от началото на файла и излишни интервали."""
    df.columns = df.columns.str.replace("ï»¿", "").str.replace("\ufeff", "").str.strip()
    return df


def add_odds(df, prefixes):
    """Взима коефициентите на букмейкърите (първите налични), ако ги има."""
    for p in prefixes:
        if {p + "H", p + "D", p + "A"} <= set(df.columns):
            df["OddsH"], df["OddsD"], df["OddsA"] = df[p + "H"], df[p + "D"], df[p + "A"]
            return df
    df["OddsH"] = df["OddsD"] = df["OddsA"] = float("nan")
    return df


def tidy(df):
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df[COLS].dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner="Свалям резултатите...")
def load_league(code):
    """Сваля данните за първенство (пазят се 6 часа, за да е бързо)."""
    if code in MAIN:
        frames = []
        for s in SEASONS:
            try:
                df = clean_columns(pd.read_csv(f"{BASE}/mmz4281/{s}/{code}.csv", encoding="latin-1"))
                frames.append(add_odds(df, ["Avg", "B365", "PS"]))
            except Exception:
                pass
        if not frames:
            return pd.DataFrame(columns=COLS)
        return tidy(pd.concat(frames, ignore_index=True))

    df = clean_columns(pd.read_csv(f"{BASE}/new/{code}.csv", encoding="latin-1"))
    df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
    df["Season"] = df["Season"].astype(str)
    last = sorted(df["Season"].unique())[-3:]
    df = add_odds(df[df["Season"].isin(last)].copy(), ["AvgC", "PSC", "B365C", "Avg", "B365"])
    return tidy(df)


@st.cache_data(ttl=3 * 3600, show_spinner="Свалям програмата...")
def load_fixtures():
    return clean_columns(pd.read_csv(f"{BASE}/fixtures.csv", encoding="latin-1"))


# ---------------- МОДЕЛ ----------------
def poisson(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


def team_strengths(df, half_life=None, ref_date=None):
    """
    Сила на атаката и защитата на всеки отбор (1.0 = средно за лигата).
    half_life: след колко дни един мач тежи наполовина (None = всички еднакво).
    """
    if half_life and len(df):
        ref_date = ref_date if ref_date is not None else df["Date"].max()
        age = (ref_date - df["Date"]).dt.days.clip(lower=0)
        w = 0.5 ** (age / half_life)
    else:
        w = pd.Series(1.0, index=df.index)

    d = pd.DataFrame({
        "HomeTeam": df["HomeTeam"], "AwayTeam": df["AwayTeam"],
        "w": w, "wh": w * df["FTHG"], "wa": w * df["FTAG"],
    })
    home_avg = d["wh"].sum() / d["w"].sum()
    away_avg = d["wa"].sum() / d["w"].sum()
    h = d.groupby("HomeTeam")[["w", "wh", "wa"]].sum()
    a = d.groupby("AwayTeam")[["w", "wh", "wa"]].sum()

    # (голове / средното + SHRINK) / (мачове + SHRINK) -> при малко мачове оценката клони към 1.0
    s = pd.DataFrame({
        "Атака дома": (h["wh"] / home_avg + SHRINK) / (h["w"] + SHRINK),
        "Защита дома": (h["wa"] / away_avg + SHRINK) / (h["w"] + SHRINK),
        "Атака навън": (a["wa"] / away_avg + SHRINK) / (a["w"] + SHRINK),
        "Защита навън": (a["wh"] / home_avg + SHRINK) / (a["w"] + SHRINK),
    })
    return s, home_avg, away_avg


def calc(model, home, away):
    """Прогноза от вече изчислени сили. Връща речник или None, ако липсват данни."""
    s, home_avg, away_avg = model
    if home not in s.index or away not in s.index or s.loc[[home, away]].isna().any().any():
        return None
    exp_h = s.loc[home, "Атака дома"] * s.loc[away, "Защита навън"] * home_avg
    exp_a = s.loc[away, "Атака навън"] * s.loc[home, "Защита дома"] * away_avg

    p1 = px = p2 = over = btts = 0
    scores = []
    for i in range(10):
        for j in range(10):
            p = poisson(i, exp_h) * poisson(j, exp_a)
            scores.append((f"{i}:{j}", p))
            if i > j:
                p1 += p
            elif i == j:
                px += p
            else:
                p2 += p
            if i + j > 2:
                over += p
            if i > 0 and j > 0:
                btts += p
    t = p1 + px + p2
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    return {
        "exp_h": exp_h, "exp_a": exp_a,
        "1": p1 / t, "X": px / t, "2": p2 / t,
        "over": over / t, "btts": btts / t,
        "top": [(r, p / t) for r, p in top],
    }


def favourite(r):
    return max(["1", "X", "2"], key=lambda k: r[k])


# ---------------- ПРОВЕРКА НА ТОЧНОСТТА ----------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def backtest(code, half_life, n_test=250):
    """
    Пуска модела върху последните n_test изиграни мача.
    За всеки ден моделът се "обучава" само с мачовете ПРЕДИ него - както в реалността.
    """
    df = load_league(code)
    if len(df) < 200:
        return None
    test = df.iloc[-n_test:]
    hits = over_hits = book_hits = book_n = n = 0
    p_actual = 0.0
    for day, games in test.groupby("Date"):
        train = df[df["Date"] < day]
        if len(train) < 150:
            continue
        model = team_strengths(train, half_life, day)
        for _, m in games.iterrows():
            r = calc(model, m["HomeTeam"], m["AwayTeam"])
            if r is None:
                continue
            actual = "1" if m["FTHG"] > m["FTAG"] else ("X" if m["FTHG"] == m["FTAG"] else "2")
            n += 1
            hits += favourite(r) == actual
            p_actual += r[actual]
            over_hits += (r["over"] > 0.5) == (m["FTHG"] + m["FTAG"] > 2)
            odds = {"1": m["OddsH"], "X": m["OddsD"], "2": m["OddsA"]}
            if all(pd.notna(v) and v > 1 for v in odds.values()):
                book_n += 1
                book_hits += min(odds, key=odds.get) == actual
    if n == 0:
        return None
    return {
        "n": n,
        "acc": hits / n,
        "p_actual": p_actual / n,
        "over_acc": over_hits / n,
        "book_acc": book_hits / book_n if book_n else None,
    }


# ---------------- САЙТ ----------------
st.set_page_config(page_title="Футболни прогнози", page_icon="⚽", layout="wide")
st.title("⚽ Футболни прогнози")
st.caption("Поасонов модел върху резултатите от последните сезони. Само за забавление, не за залози.")

with st.sidebar:
    st.header("Настройки")
    form_label = st.selectbox("Тежест на формата", list(FORM_OPTIONS), index=2,
                              help="Колко повече да тежат последните мачове спрямо по-старите. "
                                   "В раздел „Точност на модела“ виждаш коя настройка е най-точна.")
    HALF_LIFE = FORM_OPTIONS[form_label]

tab_match, tab_week, tab_table, tab_acc = st.tabs(
    ["Прогноза за мач", "Мачове тази седмица", "Сила на отборите", "Точност на модела"])

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
        recent = data[data["Date"] >= data["Date"].max() - pd.Timedelta(days=400)]
        team_list = sorted(set(recent["HomeTeam"]) | set(recent["AwayTeam"]))
        c1, c2 = st.columns(2)
        home = c1.selectbox("Домакин", team_list, index=0)
        away = c2.selectbox("Гост", team_list, index=min(1, len(team_list) - 1))

        if home == away:
            st.warning("Избери два различни отбора.")
        else:
            r = calc(team_strengths(data, HALF_LIFE), home, away)
            if r is None:
                st.warning("Няма достатъчно мачове за някой от отборите.")
            else:
                st.subheader(f"{home} - {away}")
                st.write(f"Очаквани голове: **{r['exp_h']:.2f} - {r['exp_a']:.2f}**")
                m1, m2, m3 = st.columns(3)
                m1.metric(f"1 · {home}", f"{r['1']:.0%}")
                m2.metric("X · равен", f"{r['X']:.0%}")
                m3.metric(f"2 · {away}", f"{r['2']:.0%}")
                g1, g2, g3 = st.columns(3)
                g1.metric("Над 2.5 гола", f"{r['over']:.0%}")
                g2.metric("Под 2.5 гола", f"{1 - r['over']:.0%}")
                g3.metric("Двата отбора вкарват", f"{r['btts']:.0%}")

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
        except Exception as e:
            st.error(f"Не успях да сваля програмата: {e}")
            fx = pd.DataFrame()
        if not fx.empty and "Div" not in fx.columns:
            st.error(f"Неочаквани колони във файла: {list(fx.columns)[:6]}")
        st.caption(f"Мачове в програмата: {len(fx)}")
        models, rows = {}, []
        for _, m in fx.iterrows():
            c = m.get("Div")
            if c not in MAIN:
                continue
            try:
                if c not in models:
                    models[c] = team_strengths(load_league(c), HALF_LIFE)
                r = calc(models[c], m["HomeTeam"], m["AwayTeam"])
            except Exception:
                r = None
            if r is None:
                continue
            rows.append({
                "Дата": m["Date"], "Първенство": MAIN[c],
                "Домакин": m["HomeTeam"], "Гост": m["AwayTeam"],
                "1": f"{r['1']:.0%}", "X": f"{r['X']:.0%}", "2": f"{r['2']:.0%}",
                "Фаворит": favourite(r), "Над 2.5": f"{r['over']:.0%}",
                "ДВ": f"{r['btts']:.0%}", "Резултат": r["top"][0][0],
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("ДВ = двата отбора вкарват")
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
        s, _, _ = team_strengths(d, HALF_LIFE)
        recent = d[d["Date"] >= d["Date"].max() - pd.Timedelta(days=400)]
        s = s[s.index.isin(set(recent["HomeTeam"]))]
        st.caption("Атака над 1.0 = вкарва повече от средното. Защита под 1.0 = допуска по-малко от средното.")
        st.dataframe(s.sort_values("Атака дома", ascending=False).round(2), use_container_width=True)

# --- Точност на модела ---
with tab_acc:
    st.write("Пускаме модела върху последните **250 изиграни мача**, като за всеки мач "
             "използваме само резултатите **преди** него. Така виждаме колко би бил точен в реалността.")
    code_a = st.selectbox("Първенство  ", list(ALL), format_func=lambda c: ALL[c], key="acc_league")
    if st.button("Провери точността"):
        rows = []
        book = None
        progress = st.progress(0.0, text="Проверявам...")
        for i, (label, hl) in enumerate(FORM_OPTIONS.items()):
            res = backtest(code_a, hl)
            progress.progress((i + 1) / len(FORM_OPTIONS), text=f"Проверявам: {label}")
            if res is None:
                continue
            book = res["book_acc"] if res["book_acc"] is not None else book
            rows.append({
                "Тежест на формата": label,
                "Познат изход (1/X/2)": res["acc"],
                "Среден шанс за реалния изход": res["p_actual"],
                "Познати над/под 2.5": res["over_acc"],
                "Мачове": res["n"],
            })
        progress.empty()

        if not rows:
            st.warning("Няма достатъчно данни за това първенство.")
        else:
            table = pd.DataFrame(rows)
            best = table.loc[table["Среден шанс за реалния изход"].idxmax(), "Тежест на формата"]
            show = table.copy()
            for col in ["Познат изход (1/X/2)", "Среден шанс за реалния изход", "Познати над/под 2.5"]:
                show[col] = show[col].map(lambda p: f"{p:.1%}")
            st.dataframe(show, hide_index=True, use_container_width=True)

            st.success(f"Най-добра настройка за {ALL[code_a]}: **{best}**. "
                       f"Можеш да я избереш в менюто „Настройки“ вляво.")
            if book is not None:
                st.info(f"За сравнение: фаворитът на букмейкърите е печелил в **{book:.1%}** от същите мачове.")
            st.caption("„Среден шанс за реалния изход“ е най-честната мярка: колко вероятност е дал моделът "
                       "на това, което реално се случи. По-високото е по-добре. "
                       "Познаването на 1/X/2 рядко минава 50-55%, защото футболът е непредсказуем.")
