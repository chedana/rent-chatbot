import streamlit as st
from rent_core import run_search

st.set_page_config(page_title="London RentBot", layout="centered")

st.title("London RentBot (Zoopla listings)")
st.write("用自然语言描述你的需求，例如：**Canary Wharf 附近 1 bed，预算 2400 以内，最好靠近地铁，有阳台更好**")

q = st.text_area("你的需求", height=120)

col1, col2 = st.columns([1,1])
with col1:
    run = st.button("Search")
with col2:
    st.caption("提示：可以写区域、预算、卧室数、关键词（balcony/furnished/near station）")

if run:
    if not q.strip():
        st.error("请输入需求")
    else:
        with st.spinner("Searching..."):
            try:
                ans = run_search(q.strip())
                st.subheader("推荐结果")
                st.markdown(ans)
            except Exception as e:
                st.error(f"Error: {e}")
