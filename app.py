import html
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
VALUE_MIN = 0.10  # от каква разлика спрямо букмейкърите показваме ⭐
COLS = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "OddsH", "OddsD", "OddsA"]


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
                df["Season"] = s
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


def tau(i, j, lh, la, rho):
    """
    Корекция на Dixon-Coles за ниските резултати (0:0, 1:0, 0:1, 1:1).
    Простият модел подценява равенствата; при rho < 0 те стават по-вероятни.
    """
    if i == 0 and j == 0:
        return 1 - lh * la * rho
    if i == 0 and j == 1:
        return 1 + lh * rho
    if i == 1 and j == 0:
        return 1 + la * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def expected_goals(strengths, home, away):
    s, home_avg, away_avg = strengths
    exp_h = s.loc[home, "Атака дома"] * s.loc[away, "Защита навън"] * home_avg
    exp_a = s.loc[away, "Атака навън"] * s.loc[home, "Защита дома"] * away_avg
    return exp_h, exp_a


def estimate_rho(df, strengths):
    """Намира rho, което най-добре обяснява честотата на 0:0, 1:0, 0:1 и 1:1 в първенството."""
    s = strengths[0]
    ok = df["HomeTeam"].isin(s.index) & df["AwayTeam"].isin(s.index)
    low = df[ok & (df["FTHG"] <= 1) & (df["FTAG"] <= 1)]
    if len(low) < 30:
        return 0.0
    games = [(int(m.FTHG), int(m.FTAG), *expected_goals(strengths, m.HomeTeam, m.AwayTeam))
             for m in low.itertuples()]
    best_rho, best_ll = 0.0, -float("inf")
    for k in range(-25, 16):  # rho от -0.25 до 0.15
        rho = k / 100
        vals = [tau(i, j, lh, la, rho) for i, j, lh, la in games]
        if min(vals) <= 0:
            continue
        ll = sum(math.log(v) for v in vals)
        if ll > best_ll:
            best_rho, best_ll = rho, ll
    return best_rho


def build_model(df, half_life=None, ref_date=None, use_dc=True, rho=None):
    """Модел = сили на отборите + корекция за равенства."""
    strengths = team_strengths(df, half_life, ref_date)
    if not use_dc:
        rho = 0.0
    elif rho is None:
        rho = estimate_rho(df, strengths)
    return (*strengths, rho)


def calc(model, home, away):
    """Прогноза от вече изчислен модел. Връща речник или None, ако липсват данни."""
    s, home_avg, away_avg, rho = model
    if home not in s.index or away not in s.index or s.loc[[home, away]].isna().any().any():
        return None
    exp_h, exp_a = expected_goals((s, home_avg, away_avg), home, away)

    p1 = px = p2 = over = btts = 0
    scores = []
    for i in range(10):
        for j in range(10):
            p = poisson(i, exp_h) * poisson(j, exp_a) * tau(i, j, exp_h, exp_a, rho)
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


# ---------------- КЛАСИРАНЕ И ФОРМА ----------------
ICON = {"П": "🟢", "Р": "🟡", "З": "🔴"}  # победа, равен, загуба


def team_matches(df, team):
    """Мачовете на един отбор, гледани от негова гледна точка (най-новите първи)."""
    m = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    home = m["HomeTeam"] == team
    m["Срещу"] = m["AwayTeam"].where(home, m["HomeTeam"])
    m["Къде"] = home.map({True: "дома", False: "навън"})
    m["За"] = m["FTHG"].where(home, m["FTAG"]).astype(int)
    m["Против"] = m["FTAG"].where(home, m["FTHG"]).astype(int)
    m["Изход"] = "Р"
    m.loc[m["За"] > m["Против"], "Изход"] = "П"
    m.loc[m["За"] < m["Против"], "Изход"] = "З"
    m["Точки"] = m["Изход"].map({"П": 3, "Р": 1, "З": 0})
    return m.sort_values("Date", ascending=False)


def form_icons(df, team, n=5):
    """Последните n мача като цветни точки, от най-стария към най-новия."""
    last = team_matches(df, team).head(n)["Изход"].tolist()[::-1]
    return "".join(ICON[x] for x in last)


def league_table(df):
    """Класиране за текущия сезон."""
    season = df["Season"].iloc[-1]
    cur = df[df["Season"] == season]
    rows = []
    for team in set(cur["HomeTeam"]) | set(cur["AwayTeam"]):
        m = team_matches(cur, team)
        w, d, l = (m["Изход"] == "П").sum(), (m["Изход"] == "Р").sum(), (m["Изход"] == "З").sum()
        gf, ga = m["За"].sum(), m["Против"].sum()
        rows.append({"Отбор": team, "М": len(m), "П": w, "Р": d, "З": l,
                     "Голове": f"{gf}:{ga}", "ГР": gf - ga, "Т": 3 * w + d,
                     "_gf": gf, "Форма": form_icons(cur, team)})
    t = pd.DataFrame(rows).sort_values(["Т", "ГР", "_gf"], ascending=False).drop(columns="_gf")
    t.insert(0, "#", range(1, len(t) + 1))
    return t, season


def last_matches_table(df, team, n=5):
    m = team_matches(df, team).head(n)
    return pd.DataFrame({
        "Дата": m["Date"].dt.strftime("%d.%m.%Y"),
        "Срещу": m["Срещу"],
        "Къде": m["Къде"],
        "Резултат": m["За"].astype(str) + ":" + m["Против"].astype(str),
        "": m["Изход"].map(ICON),
    })


def head_to_head(df, a, b, n=6):
    m = df[((df["HomeTeam"] == a) & (df["AwayTeam"] == b)) |
           ((df["HomeTeam"] == b) & (df["AwayTeam"] == a))].sort_values("Date", ascending=False).head(n)
    return pd.DataFrame({
        "Дата": m["Date"].dt.strftime("%d.%m.%Y"),
        "Домакин": m["HomeTeam"],
        "Резултат": m["FTHG"].astype(int).astype(str) + ":" + m["FTAG"].astype(int).astype(str),
        "Гост": m["AwayTeam"],
    })


# ---------------- ПРОВЕРКА НА ТОЧНОСТТА ----------------
def implied_probs(oh, od, oa):
    """Вероятностите, които букмейкърите влагат в коефициентите (без тяхната печалба)."""
    if not all(pd.notna(v) and v > 1 for v in (oh, od, oa)):
        return None
    raw = {"1": 1 / oh, "X": 1 / od, "2": 1 / oa}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def backtest(code, half_life, use_dc, n_test=250):
    """
    Пуска модела върху последните n_test изиграни мача.
    За всеки ден моделът се "обучава" само с мачовете ПРЕДИ него - както в реалността.
    """
    df = load_league(code)
    if len(df) < 200:
        return None
    test = df.iloc[-n_test:]
    # корекцията за равенства се изчислява веднъж, от мачовете преди тестовия период
    before = df[df["Date"] < test["Date"].iloc[0]]
    rho = build_model(before, half_life, use_dc=use_dc)[3] if len(before) >= 150 else 0.0

    hits = over_hits = n = 0
    p_actual = 0.0
    book = {"n": 0, "hits": 0, "p_actual": 0.0}
    for day, games in test.groupby("Date"):
        train = df[df["Date"] < day]
        if len(train) < 150:
            continue
        model = build_model(train, half_life, day, use_dc=use_dc, rho=rho)
        for _, m in games.iterrows():
            r = calc(model, m["HomeTeam"], m["AwayTeam"])
            if r is None:
                continue
            actual = "1" if m["FTHG"] > m["FTAG"] else ("X" if m["FTHG"] == m["FTAG"] else "2")
            n += 1
            hits += favourite(r) == actual
            p_actual += r[actual]
            over_hits += (r["over"] > 0.5) == (m["FTHG"] + m["FTAG"] > 2)
            bp = implied_probs(m["OddsH"], m["OddsD"], m["OddsA"])
            if bp:
                book["n"] += 1
                book["hits"] += max(bp, key=bp.get) == actual
                book["p_actual"] += bp[actual]
    if n == 0:
        return None
    return {
        "n": n,
        "acc": hits / n,
        "p_actual": p_actual / n,
        "over_acc": over_hits / n,
        "rho": rho,
        "book_n": book["n"],
        "book_acc": book["hits"] / book["n"] if book["n"] else None,
        "book_p_actual": book["p_actual"] / book["n"] if book["n"] else None,
    }


# ---------------- ДИЗАЙН ----------------
CSS = """<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg, #14532d 0%, #0b1220 75%); border: 1px solid #1f2a44;
       border-radius: 16px; padding: 22px 26px; margin-bottom: 1rem;}
.hero h1 {margin: 0; padding: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .35rem 0 0; color: #9ca3af;}
.match-card {background: #131c2e; border: 1px solid #1f2a44; border-radius: 16px; padding: 22px; margin: 1rem 0;}
.teams {display: flex; justify-content: space-between; align-items: center; gap: 12px;}
.team {flex: 1; font-size: 1.4rem; font-weight: 700; color: #fff;}
.team.away {text-align: right;}
.xg {text-align: center;}
.xg .v {font-size: 1.6rem; font-weight: 800; color: #22c55e; white-space: nowrap;}
.xg .l {color: #9ca3af; font-size: .75rem;}
.bar {display: flex; height: 38px; border-radius: 10px; overflow: hidden; margin-top: 18px;
      font-weight: 700; color: #fff; font-size: .9rem;}
.bar div {display: flex; align-items: center; justify-content: center;}
.b1 {background: #16a34a;} .bx {background: #4b5563;} .b2 {background: #2563eb;}
.legend {display: flex; justify-content: space-between; color: #9ca3af; font-size: .8rem; margin-top: 6px;}
.stats {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px;}
.stat {background: #0b1220; border: 1px solid #1f2a44; border-radius: 12px; padding: 12px; text-align: center;}
.stat .v {font-size: 1.4rem; font-weight: 800; color: #fff;}
.stat .l {color: #9ca3af; font-size: .8rem;}
@media (max-width: 640px) {
  .team {font-size: 1.05rem;} .xg .v {font-size: 1.2rem;} .stat .v {font-size: 1.1rem;}
}
</style>"""


def match_card(home, away, r):
    """Карта на мача: имена, очаквани голове, цветна лента 1/X/2 и голове."""
    h, a = html.escape(home), html.escape(away)

    def seg(cls, p):
        label = f"{p:.0%}" if p >= 0.08 else ""
        return f'<div class="{cls}" style="width:{p * 100:.1f}%">{label}</div>'

    return "".join([
        '<div class="match-card">',
        '<div class="teams">',
        f'<div class="team">{h}</div>',
        f'<div class="xg"><div class="v">{r["exp_h"]:.1f} : {r["exp_a"]:.1f}</div>'
        '<div class="l">очаквани голове</div></div>',
        f'<div class="team away">{a}</div>',
        '</div>',
        '<div class="bar">', seg("b1", r["1"]), seg("bx", r["X"]), seg("b2", r["2"]), '</div>',
        f'<div class="legend"><span>1 · победа {h}</span><span>X · равен</span><span>2 · победа {a}</span></div>',
        '<div class="stats">',
        f'<div class="stat"><div class="v">{r["over"]:.0%}</div><div class="l">Над 2.5 гола</div></div>',
        f'<div class="stat"><div class="v">{1 - r["over"]:.0%}</div><div class="l">Под 2.5 гола</div></div>',
        f'<div class="stat"><div class="v">{r["btts"]:.0%}</div><div class="l">Двата вкарват</div></div>',
        '</div></div>',
    ])


PCT = lambda label: st.column_config.ProgressColumn(label, format="%d%%", min_value=0, max_value=100)

# ---------------- САЙТ ----------------
st.set_page_config(page_title="Футболни прогнози", page_icon="⚽", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>⚽ Футболни прогнози</h1>'
            '<p>Поасонов модел върху резултатите от последните сезони · 32 първенства · '
            'само за забавление, не за залози</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Настройки")
    form_label = st.selectbox("Тежест на формата", list(FORM_OPTIONS), index=2,
                              help="Колко повече да тежат последните мачове спрямо по-старите. "
                                   "В раздел „Точност на модела“ виждаш коя настройка е най-точна.")
    HALF_LIFE = FORM_OPTIONS[form_label]
    USE_DC = st.toggle("Корекция за равенства", value=True,
                       help="Метод на Dixon-Coles: простият модел подценява резултати като 0:0 и 1:1. "
                            "Корекцията се изчислява от данните на всяко първенство поотделно.")
    st.divider()
    st.caption("Данни: football-data.co.uk. Резултатите се обновяват няколко пъти седмично, "
               "след изиграните кръгове.")

tab_match, tab_week, tab_standings, tab_table, tab_acc = st.tabs(
    ["Прогноза за мач", "Мачове тази седмица", "Класиране", "Сила на отборите", "Точност на модела"])

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
            r = calc(build_model(data, HALF_LIFE, use_dc=USE_DC), home, away)
            if r is None:
                st.warning("Няма достатъчно мачове за някой от отборите.")
            else:
                st.markdown(match_card(home, away, r), unsafe_allow_html=True)

                st.write("**Най-вероятни резултати**")
                top = pd.DataFrame(r["top"], columns=["Резултат", "Вероятност"])
                top["Вероятност"] = (top["Вероятност"] * 100).round()
                st.dataframe(top, hide_index=True, use_container_width=False,
                             column_config={"Вероятност": PCT("Вероятност")})

            # --- Форма на двата отбора ---
            st.divider()
            st.subheader("Форма в последните 5 мача")
            f1, f2 = st.columns(2)
            for col, team in ((f1, home), (f2, away)):
                col.markdown(f"**{team}** {form_icons(data, team)}")
                col.dataframe(last_matches_table(data, team), hide_index=True, use_container_width=True)

            st.write("**Точки в последните 10 мача** (натрупани)")
            chart = pd.DataFrame({
                t: team_matches(data, t).head(10)["Точки"].iloc[::-1].cumsum().reset_index(drop=True)
                for t in (home, away)
            })
            chart.index = chart.index + 1
            st.line_chart(chart, x_label="Мач", y_label="Точки")

            # --- Последни срещи ---
            st.subheader("Последни срещи между двата отбора")
            h2h = head_to_head(data, home, away)
            if h2h.empty:
                st.info("Няма срещи между тези отбори в последните сезони.")
            else:
                st.dataframe(h2h, hide_index=True, use_container_width=False)

# --- Класиране ---
with tab_standings:
    code_s = st.selectbox("Първенство", list(ALL), format_func=lambda c: ALL[c], key="standings_league")
    try:
        ds = load_league(code_s)
    except Exception:
        ds = pd.DataFrame()
    if ds.empty:
        st.error("Не успях да сваля данни за това първенство.")
    else:
        table, season = league_table(ds)
        st.caption(f"Сезон {season} · последен изигран мач: {ds['Date'].max():%d.%m.%Y} · "
                   f"Форма: последните 5 мача, отляво надясно към най-новия (🟢 победа, 🟡 равен, 🔴 загуба)")
        st.dataframe(table, hide_index=True, use_container_width=True, height=35 * (len(table) + 1) + 3)

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
        if not fx.empty:
            fx = add_odds(fx, ["Avg", "B365", "PS"])
        models, rows = {}, []
        for _, m in fx.iterrows():
            c = m.get("Div")
            if c not in MAIN:
                continue
            try:
                if c not in models:
                    models[c] = build_model(load_league(c), HALF_LIFE, use_dc=USE_DC)
                r = calc(models[c], m["HomeTeam"], m["AwayTeam"])
            except Exception:
                r = None
            if r is None:
                continue

            # сравнение с букмейкърите: къде моделът вижда по-голям шанс от тях
            odds = {"1": m["OddsH"], "X": m["OddsD"], "2": m["OddsA"]}
            has_odds = all(pd.notna(v) and v > 1 for v in odds.values())
            value = ""
            if has_odds:
                edge = {k: r[k] * odds[k] - 1 for k in odds}
                k = max(edge, key=edge.get)
                if edge[k] >= VALUE_MIN:
                    value = f"⭐ {k} (+{edge[k]:.0%})"

            rows.append({
                "Дата": m["Date"], "Първенство": MAIN[c],
                "Домакин": m["HomeTeam"], "Гост": m["AwayTeam"],
                "1": round(r["1"] * 100), "X": round(r["X"] * 100), "2": round(r["2"] * 100),
                "Коеф. 1 / X / 2": (f"{odds['1']:.2f} / {odds['X']:.2f} / {odds['2']:.2f}"
                                    if has_odds else "—"),
                "Стойност": value,
                "Над 2.5": round(r["over"] * 100),
                "ДВ": round(r["btts"] * 100), "Резултат": r["top"][0][0],
            })
        if rows:
            out = pd.DataFrame(rows)
            if st.toggle("Само мачове със ⭐"):
                out = out[out["Стойност"] != ""]
            st.dataframe(out, hide_index=True, use_container_width=True,
                         column_config={k: PCT(k) for k in ["1", "X", "2", "Над 2.5", "ДВ"]})
            st.caption(f"ДВ = двата отбора вкарват. ⭐ = моделът дава поне {VALUE_MIN:.0%} повече шанс, "
                       "отколкото излиза от средния коефициент на букмейкърите. Това е сравнение, "
                       "не съвет за залог: букмейкърите обикновено са по-точни от прост модел.")
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
        rows, book, rho = [], None, None
        runs = [(label, hl, dc) for label, hl in FORM_OPTIONS.items() for dc in (False, True)]
        progress = st.progress(0.0, text="Проверявам...")
        for i, (label, hl, dc) in enumerate(runs):
            progress.progress(i / len(runs), text=f"Проверявам: {label}")
            res = backtest(code_a, hl, dc)
            if res is None:
                continue
            if dc:
                rho = res["rho"]
            if res["book_acc"] is not None:
                book = res
            rows.append({
                "Модел": label + (" + корекция за равенства" if dc else ""),
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
            best = table.loc[table["Среден шанс за реалния изход"].idxmax()]
            if book is not None:
                table = pd.concat([table, pd.DataFrame([{
                    "Модел": "📊 Букмейкърите (среден коефициент)",
                    "Познат изход (1/X/2)": book["book_acc"],
                    "Среден шанс за реалния изход": book["book_p_actual"],
                    "Познати над/под 2.5": None,
                    "Мачове": book["book_n"],
                }])], ignore_index=True)
            show = table.copy()
            for col in ["Познат изход (1/X/2)", "Среден шанс за реалния изход", "Познати над/под 2.5"]:
                show[col] = show[col].map(lambda p: "—" if pd.isna(p) else f"{p:.1%}")
            st.dataframe(show, hide_index=True, use_container_width=True)

            st.success(f"Най-добра настройка за {ALL[code_a]}: **{best['Модел']}**. "
                       f"Можеш да я избереш в менюто „Настройки“ вляво.")
            if book is not None:
                diff = best["Среден шанс за реалния изход"] - book["book_p_actual"]
                verdict = ("Моделът ни е **по-точен** от букмейкърите в тези мачове. Рядко се случва!"
                           if diff > 0 else
                           f"Букмейкърите са с **{-diff:.1%}** по-точни. Нормално е: те ползват много повече "
                           "информация (контузии, състави, пазара).")
                st.info(verdict)
            if rho is not None:
                st.caption(f"Корекция за равенства в това първенство: rho = {rho:+.2f} "
                           "(отрицателно = равенствата 0:0 и 1:1 са по-чести, отколкото предвижда простият модел).")
            st.caption("„Среден шанс за реалния изход“ е най-честната мярка: колко вероятност е дал моделът "
                       "на това, което реално се случи. По-високото е по-добре. "
                       "Познаването на 1/X/2 рядко минава 50-55%, защото футболът е непредсказуем.")
