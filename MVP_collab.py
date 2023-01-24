import csv
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
from plotly import offline
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from catboost import CatBoost,Pool
from catboost import CatBoostClassifier

import shap
import pickle

import os

st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
  st.header("離反顧客分析")

  name_list = ["AAA"]
  login_name = st.sidebar.text_input("Loginname")

  if login_name in name_list:
    #データの前処理
    df = pd.read_csv("./telecom_customer_churn.csv")
    df.columns=df.columns.str.replace(" ","").str.lower()
    df.avgmonthlylongdistancecharges=df.avgmonthlylongdistancecharges.fillna(0.0)
    df.multiplelines=df.multiplelines.fillna('no phone service')
    no_internet=['internettype','onlinesecurity','onlinebackup','deviceprotectionplan','premiumtechsupport','streamingtv',
                'streamingmovies','streamingmusic','unlimiteddata']
    df[no_internet]=df[no_internet].fillna('no internet service')
    df.avgmonthlygbdownload=df.avgmonthlygbdownload.fillna(0)

    customer_id = df["customerid"]

    df=df.drop(columns=['customerid','churncategory','churnreason','totalrefunds','zipcode','longitude','latitude','city'])
    df=df.loc[~df.customerstatus.str.contains('Join')]
    df.reset_index(drop=True,inplace=True)

    menu = ["離反割合", "契約プラン別", "離反確率分布", "確率Simulator", "離反Factor"]

    chosen_menu = st.sidebar.selectbox(
            '離反顧客分析の種類を選択：',menu
        )

    modelfile = 'trained_model.pkl'
    is_modelfile = os.path.isfile(modelfile)

    allfile = "all.csv"
    xtrainfile = "X_train.csv"
    xtestfile = "X_test.csv"
    ytrainfile = "y_train.csv"
    ytestfile = "y_test.csv"
    is_allfile = os.path.isfile(allfile)
    is_xtrainfile = os.path.isfile(xtrainfile)
    is_xtestfile = os.path.isfile(xtestfile)
    is_ytrainfile = os.path.isfile(ytrainfile)
    is_ytestfile = os.path.isfile(ytestfile)
    
    no_update = True

    print(is_allfile)
    print(is_xtrainfile)
    print(is_xtestfile)
    print(is_ytrainfile)
    print(is_ytestfile)
    print(no_update)


    if is_allfile and is_xtrainfile and is_xtestfile and is_ytrainfile and is_ytestfile and no_update:
        X = pd.read_csv("all.csv")
        X_train = pd.read_csv("X_train.csv")
        X_test = pd.read_csv("X_test.csv")
        y_train = pd.read_csv("y_train.csv",header=None)
        y_test = pd.read_csv("y_test.csv",header=None)
        print("データを読み込みました")

    else:
        le = LabelEncoder()
        le_count = 0
        for col in df.columns[1:]: #genderは個別に実施
            if df[col].dtype == 'object':
                if len(list(df[col].unique())) <= 2:
                    le.fit(df[col])
                    df[col] = le.transform(df[col])
                    le_count += 1
        df['gender'] = [1 if each == 'Female' else 0 for each in df['gender']]

        def encode_data(dataframe):
            if dataframe.dtype == "object":
                dataframe = LabelEncoder().fit_transform(dataframe)
            return dataframe

        #残りのものも適用
        data = df.apply(lambda x: encode_data(x))

        X = data.drop(columns = "customerstatus")
        y = data["customerstatus"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4, stratify =y)

        col=['totalcharges','avgmonthlylongdistancecharges','monthlycharge','totalrevenue','totallongdistancecharges',
            'tenureinmonths','totallongdistancecharges','totalextradatacharges']  


        scaler = StandardScaler()
        X_train[col] = StandardScaler().fit_transform(X_train[col])
        X_test[col] = StandardScaler().fit_transform(X_test[col])

        
        X.to_csv(allfile, index=None)
        X_train.to_csv(xtrainfile,index=None)
        X_test.to_csv(xtestfile,index=None)
        np.savetxt(ytrainfile, y_train)
        np.savetxt(ytestfile, y_test) 
        #y_test.to_csv("y_test.csv",index=None)
        print("データを作成しました")

    if is_modelfile:
        cat_model = pickle.load(open('trained_model.pkl', 'rb'))
        print("モデルが読み込まれました")
    else:
        cat_model = CatBoostClassifier (random_state = 42, eval_metric = 'AUC', verbose = 0)
        cat_model.fit(X_train, y_train, eval_set = [(X_test,y_test)])
        pickle.dump(cat_model, open(modelfile, 'wb'))
        
        print("モデル作成されました")

    
    prob = cat_model.predict(X,prediction_type='Probability')[:,1]


    if chosen_menu == "離反割合":
        type_ = ["No", "yes"]
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Pie(labels=type_, values=df['customerstatus'].value_counts(), name="顧客Status"))

        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)

        fig.update_layout(
            title_text="<b>Churn割合</b>",
            # Add annotations in the center of the donut pies.
            annotations=[dict(text='Churn', x=0.5, y=0.5, font_size=20, showarrow=False)])
        st.plotly_chart(fig)

    elif chosen_menu == "契約プラン別":
        fig = px.histogram(df, x="customerstatus", color = "contract", barmode = "group", title = "<b>契約プラン別分布<b>")
        fig.update_layout(width=700, height=500, bargap=0.2)
        st.plotly_chart(fig)

    elif chosen_menu == "離反確率分布":
        df_tem = pd.DataFrame([customer_id, prob]).T
        df_tem.columns = ["id", "継続確率"]
        st.dataframe(df_tem)
        
    elif chosen_menu == "確率Simulator":
        st.write("分析したい顧客の属性データをアップロードしてください")

        csvfile = st.file_uploader("CSVファイルのアップロード", type="csv")
        target_data = pd.read_csv(csvfile)
        st.subheader("顧客属性")
        st.dataframe(target_data)
        
        proba = pd.DataFrame(cat_model.predict(target_data,prediction_type='Probability')[:,:])
        proba.columns = ["churn", "非Churn"]
        st.subheader("予測結果")
        st.write(proba)
        st.subheader("アップロード頂いた顧客がChurnする確率は{:.2f}%です。".format(proba.iloc[0,0]*100))
        st.write("高い場合には、アクションをとっていくことをおすすめします。")

    elif chosen_menu == "離反Factor":
        st.write("真ん中より左がChurnで右が継続(非Churn)を表す")
        explainer = shap.TreeExplainer(cat_model)
        pool = Pool(X_train, y_train)
        shap_values = explainer.shap_values(pool)

        st.pyplot(shap.summary_plot(shap_values, X_train))

if __name__ == "__main__":
    main()
      