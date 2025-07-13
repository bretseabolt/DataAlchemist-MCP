import streamlit as st

main_page = st.Page(
    "da_pages/main_page.py",
    title="Data Alchemist",
    default=True,
    icon="🧪"
)
dashboard_page = st.Page(
    "da_pages/data_viewer.py",
    title="Alchemy Dashboard",
    icon="📊"
)

pages = [main_page, dashboard_page]

pg = st.navigation(pages)
pg.run()