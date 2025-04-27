import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="📚 Community Data Chatbot", layout="centered")

# ========== 📥 Step 1: Load all data ==========
@st.cache_data

def load_data():
    indexes_dataset = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/indexes_dataset.csv")
    metadata_with_index = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/metadata_with_index.csv")
    indexes_metadata = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/indexes_metadata.csv")
    manual_mapping = pd.read_csv("E:/PythonCode/BADM550_NEW_PROJECT/data/Completed_FineNeighborhood_to_CommunityAreaName.csv")
    with open("E:/PythonCode/BADM550_NEW_PROJECT/data/Hotmapdataset.json", "r", encoding="utf-8") as f:
        hotmap_data = json.load(f)
    return indexes_dataset, metadata_with_index, indexes_metadata, manual_mapping, hotmap_data

indexes_dataset, metadata_with_index, indexes_metadata, manual_mapping, hotmap_data = load_data()

# ========== 🧠 Step 2: Smart Search Function ==========
def search_answer(user_input):
    user_input_lower = user_input.lower()

    # 1. Check big index definition
    for _, row in indexes_metadata.iterrows():
        if row['index_name'].lower() in user_input_lower:
            return f"📈 {row['index_name']} 是: {row['index_description']}"

    # 2. Check feature definition
    for _, row in metadata_with_index.iterrows():
        if row['column_name'].lower() in user_input_lower:
            return f"🔎 {row['column_name']} 的定义是: {row['description']}"

    # 3. Check neighborhood mapping
    for _, row in manual_mapping.iterrows():
        if row['FineNeighborhood'].lower() in user_input_lower:
            return f"🏙️ {row['FineNeighborhood']} 属于 {row['CommunityAreaName']} 社区。"

    # 4. Check indexes values
    for neighborhood in indexes_dataset['FineNeighborhood'].dropna().unique():
        if neighborhood.lower() in user_input_lower:
            matched_row = indexes_dataset[indexes_dataset['FineNeighborhood'].str.lower() == neighborhood.lower()].iloc[0]
            index_cols = [col for col in matched_row.index if 'Index' in col or 'EVI' in col or 'SSI' in col or 'EOI' in col]
            parts = [f"{col}: {matched_row[col]:.2f}" for col in index_cols]
            return f"📊 {neighborhood} 的主要指数如下：\n" + "\n".join(parts)

    # 5. Check hotmap risk factors
    for record in hotmap_data:
        if record['CommunityAreaName'].lower() in user_input_lower:
            parts = [f"{k}: {v:.2f}" for k, v in record.items() if k != 'CommunityAreaName']
            return f"🔥 {record['CommunityAreaName']} 的风险因子评分如下：\n" + "\n".join(parts)

    return "❓ 抱歉，我没有在数据中找到相关信息，请换种提问方式试试。"

# ========== 🖥️ Step 3: Streamlit Frontend ==========
st.title("📚 Community Data Chatbot")

st.markdown("""
<style>
.chat-bubble {
  background-color: #f0f2f6;
  padding: 12px;
  border-radius: 10px;
  margin-bottom: 10px;
  display: inline-block;
  max-width: 90%;
}
.user-bubble {
  background-color: #d1e7dd;
  text-align: right;
  margin-left: auto;
}
</style>
""", unsafe_allow_html=True)

user_input = st.text_input("请输入你的问题:", key="user_input")

if user_input:
    st.markdown(f'<div class="chat-bubble user-bubble">{user_input}</div>', unsafe_allow_html=True)
    response = search_answer(user_input)
    st.markdown(f'<div class="chat-bubble">{response}</div>', unsafe_allow_html=True)
