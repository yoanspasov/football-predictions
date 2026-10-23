import pandas as pd
import streamlit as st

from core import (ALL, FORM_OPTIONS, MAIN, VALUE_MIN, PCT, add_odds, backtest, build_model, calc,
                  favourite, form_icons, head_to_head, last_matches_table, league_table, load_fixtures,
                  load_league, match_card, settings, team_matches, team_strengths)

# Страница „Прогнози“: досегашните пет раздела.
HALF_LIFE, USE_DC = settings()

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
