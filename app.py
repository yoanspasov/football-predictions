import streamlit as st

from core import CSS, FORM_OPTIONS

# Главният файл: дизайн, настройки в менюто вляво и навигация между страниците.
st.set_page_config(page_title="Футболни прогнози", page_icon="⚽", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>⚽ Футболни прогнози</h1>'
            '<p>Поасонов модел върху резултатите от последните сезони · 32 първенства · '
            'само за забавление, не за залози</p></div>', unsafe_allow_html=True)

pages = st.navigation([
    st.Page("home.py", title="Прогнози", icon="⚽", default=True),
    st.Page("favorites.py", title="Моите отбори", icon="⭐"),
])

with st.sidebar:
    st.header("Настройки")
    form_label = st.selectbox("Тежест на формата", list(FORM_OPTIONS), index=2,
                              help="Колко повече да тежат последните мачове спрямо по-старите. "
                                   "В раздел „Точност на модела“ виждаш коя настройка е най-точна.")
    st.session_state["half_life"] = FORM_OPTIONS[form_label]
    st.session_state["use_dc"] = st.toggle(
        "Корекция за равенства", value=True,
        help="Метод на Dixon-Coles: простият модел подценява резултати като 0:0 и 1:1. "
             "Корекцията се изчислява от данните на всяко първенство поотделно.")
    st.divider()
    st.caption("Данни: football-data.co.uk. Резултатите се обновяват няколко пъти седмично, "
               "след изиграните кръгове.")

pages.run()
