import pandas as pd
import streamlit as st

from core import (ALL, MAIN, build_model, calc, form_icons, last_matches_table, league_table,
                  load_fixtures, match_card, safe_league, settings)

# Страница „Моите отбори“: любими отбори с място, форма и прогноза за следващия мач.
HALF_LIFE, USE_DC = settings()

st.subheader("⭐ Моите отбори")
st.caption("Списъкът се пази в адреса на страницата. Добави я в отметките на браузъра (bookmark) "
           "и любимите ти отбори ще са там всеки път.")

# Любимите отбори се пазят като "код|отбор", например "E0|Arsenal"
if "favs" not in st.session_state:
    saved = st.query_params.get_all("fav")
    st.session_state.favs = [f for f in saved if "|" in f and f.split("|", 1)[0] in ALL]

# --- Добавяне ---
with st.expander("➕ Добави отбор", expanded=not st.session_state.favs):
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    code = c1.selectbox("Първенство", list(ALL), format_func=lambda c: ALL[c], key="fav_league")
    data = safe_league(code)
    if data.empty:
        teams = []
    else:
        recent = data[data["Date"] >= data["Date"].max() - pd.Timedelta(days=400)]
        teams = sorted(set(recent["HomeTeam"]) | set(recent["AwayTeam"]))
    team = c2.selectbox("Отбор", teams, key="fav_team")
    if c3.button("Добави", use_container_width=True, disabled=not teams):
        key = f"{code}|{team}"
        if key not in st.session_state.favs:
            st.session_state.favs.append(key)

# записваме списъка в адреса на страницата
if st.session_state.favs:
    st.query_params["fav"] = st.session_state.favs
else:
    st.query_params.pop("fav", None)
    st.info("Още нямаш любими отбори. Избери първенство и отбор отгоре.")

# --- Програмата за следващите мачове (само за главните първенства) ---
try:
    fx = load_fixtures()
except Exception:
    fx = pd.DataFrame(columns=["Div", "Date", "HomeTeam", "AwayTeam"])

models = {}
for key in list(st.session_state.favs):
    code, team = key.split("|", 1)
    data = safe_league(code)

    with st.container(border=True):
        head, remove = st.columns([5, 1], vertical_alignment="center")
        head.markdown(f"### {team}\n{ALL[code]}")
        if remove.button("✖ Махни", key=f"rm_{key}", use_container_width=True):
            st.session_state.favs.remove(key)
            st.rerun()

        if data.empty or team not in set(data["HomeTeam"]) | set(data["AwayTeam"]):
            st.warning("Няма данни за този отбор.")
            continue

        # място в класирането и форма
        table, season = league_table(data)
        row = table[table["Отбор"] == team]
        m1, m2, m3, m4 = st.columns(4)
        if not row.empty:
            r0 = row.iloc[0]
            m1.metric("Място", f"{int(r0['#'])}. от {len(table)}")
            m2.metric("Точки", int(r0["Т"]), help=f"{int(r0['М'])} мача")
            m3.metric("Голове", r0["Голове"])
        m4.metric("Форма (последни 5)", form_icons(data, team) or "—")

        # следващ мач и прогноза
        nxt = None
        if code in MAIN and "Div" in fx.columns:
            games = fx[(fx["Div"] == code) & ((fx["HomeTeam"] == team) | (fx["AwayTeam"] == team))]
            if not games.empty:
                nxt = games.iloc[0]
        if nxt is not None:
            if code not in models:
                models[code] = build_model(data, HALF_LIFE, use_dc=USE_DC)
            r = calc(models[code], nxt["HomeTeam"], nxt["AwayTeam"])
            st.markdown(f"**Следващ мач:** {nxt['Date']}")
            if r is not None:
                st.markdown(match_card(nxt["HomeTeam"], nxt["AwayTeam"], r), unsafe_allow_html=True)
            else:
                st.write(f"{nxt['HomeTeam']} - {nxt['AwayTeam']} (няма достатъчно данни за прогноза)")
        elif code in MAIN:
            st.caption("Следващият мач още не е в програмата. Обикновено се появява в началото на седмицата.")
        else:
            st.caption("За това първенство нямаме програма, само резултати.")

        with st.expander("Последни 5 мача"):
            st.dataframe(last_matches_table(data, team), hide_index=True, use_container_width=True)
