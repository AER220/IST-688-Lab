import streamlit as st

lab1 = st.Page("LAB1.py", title="Lab 1", icon="1️⃣")
lab2 = st.Page("LAB2.py", title="Lab 2", icon="2️⃣")
lab3 = st.Page("LAB3.py", title="Lab 3", icon="3️⃣")
lab4 = st.Page("LAB4.py", title="Lab 4", icon="4️⃣", default=True)

pg = st.navigation([lab1, lab2, lab3, lab4])
pg.run()