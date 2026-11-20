import streamlit as st

lab1 = st.Page("LAB1.py", title="Lab 1", icon="1️⃣")
lab2 = st.Page("LAB2.py", title="Lab 2", icon="2️⃣")
lab3 = st.Page("LAB3.py", title="Lab 3", icon="3️⃣")
lab4 = st.Page("LAB4.py", title="Lab 4", icon="4️⃣")
lab5 = st.Page("LAB5.py", title="Lab 5", icon="5️⃣")
lab6 = st.Page("LAB6.py", title="Lab 6", icon="6️⃣")
lab8 = st.Page("LAB8.py", title="Lab 8", icon="8️⃣")
lab9 = st.Page("LAB9.py", title="Lab 9", icon="9️⃣")
lab9agents = st.Page("LAB9_agents.py", title="Lab 9 Agents", icon="🤝", default=True)

pg = st.navigation([lab1, lab2, lab3, lab4, lab5, lab6, lab8, lab9, lab9agents])
pg.run()