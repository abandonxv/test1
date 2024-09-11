import streamlit as st

# 设置标题
st.title("Streamlit 应用示例")

# 创建一个输入框，让用户输入文本
user_input = st.text_input("请输入一些文字:")

# 如果用户输入了文本，显示输出
if user_input:
    st.write("你输入的内容是:", user_input)

# 显示一个简单的按钮
if st.button('点击我'):
    st.write('按钮已点击！')

# 添加一个数字滑块
number = st.slider('选择一个数字', 0, 100, 50)
st.write('你选择的数字是:', number)

# 生成一个简单的折线图
import numpy as np
import pandas as pd

data = pd.DataFrame({
    'x': np.arange(1, 101),
    'y': np.random.randn(100).cumsum()
})

st.line_chart(data)
