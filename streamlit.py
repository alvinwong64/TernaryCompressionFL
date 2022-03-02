import streamlit as st
import pandas as pd
import plotly.express as px

base_df = pd.read_excel(r'..\UM\Sem 7\FYP2\Results.xlsx', sheet_name='Baseline', index_col=0)
ternary_df = pd.read_excel(r'..\UM\Sem 7\FYP2\Results.xlsx', sheet_name='Ternary', index_col=0)
compression_df = pd.read_excel(r'..\UM\Sem 7\FYP2\Results.xlsx', sheet_name="Compression", index_col=0)
st.title('Ternary Compression in Federated Learning ')

st.markdown("""
Showcase of the experiment results of the ternary compression. \n
All of these models are trained using CIFAR 10 dataset\n
The fixed training hyperparameters are set as follows:\n
\n
* ** Learning rate   **     : 0.01
* ** Learning rate Decay ** : Decay by 0.1 every 50 epochs
* ** Epochs          **     : 100
* ** Batch Size      **     : 128
* ** Weight Decay    **     : 0.001
* ** Number of Clients **    : 10
* ** Loss Function  **      : Cross Entropy Loss
* ** Optimazation Algorithm ** : SGD with momentum of 0.9
\n
The experiment consist of 4 conditions, which are:\n
* ** local **: Non federated training, which means it just runs like a normal deep learning training, used as a control experiment
* ** iid  ** : Federated training, the data is evenly distributed among the 10 clients
* ** non-iid(5 classes)  ** : Federated training, the data is unevenly distributed among the clients, each client only receive the subset of the dataset, which has only 5 classes
* ** non-iid(2 classes)  ** : Federated training, the data is unevenly distributed among the clients, each client only receive the subset of the dataset, which has only 2 classes
""")

st.sidebar.header('User Input Features')

select_model = sorted(base_df.index)
selected_model = st.sidebar.multiselect('Models', select_model, select_model)

unique_condition = base_df.columns
selected_condition = st.sidebar.multiselect('Condition', unique_condition.tolist(), unique_condition.tolist())

# Filtering data
selected_base_df = base_df[selected_condition][base_df.index.isin(selected_model)]

st.header('Baseline (Non Ternary Model) Results')
st.write('This shows the baseline of the experiment results')
st.dataframe(selected_base_df)

fig = px.bar(selected_base_df, x=selected_base_df.index, y=selected_base_df.columns, barmode='group', height=400)
st.plotly_chart(fig)

selected_ternary_df = ternary_df[selected_condition][ternary_df.index.isin(selected_model)]

st.header('Ternary Model Results')
st.write('This shows the experiment results of ternary model')
st.dataframe(selected_ternary_df)

fig = px.bar(selected_ternary_df, x=selected_ternary_df.index, y=selected_base_df.columns, barmode='group', height=400)
st.plotly_chart(fig)

st.header('Compression Results')
st.write('This shows the compression fize size of the model using Huffman Coding')
st.dataframe(compression_df[ternary_df.index.isin(selected_model)])


