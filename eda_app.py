import streamlit as st
import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import datasets



def load_data(path):
    # CSV 파일 가져올 시
    df = pd.read_csv(path)

    # DB에서 가져올 경우

    # 크롤링해서 가져올 경우
    return df


def run_eda_app():
    st.subheader('탐색적 자료 분석')
    with st.expander('데이터셋 정보'):
        st.markdown(utils.attrib_info)

    # 데이터셋 불러오기
    DATA_PATH = 'data/iris.csv'
    iris_df = load_data(DATA_PATH)

    # 서브메뉴 지정
    submenu = st.sidebar.selectbox("서브메뉴", ['기술통계량','그래프'])
    if submenu == '기술통계량':
        st.dataframe(iris_df)

        with st.expander('데이터 타입'):
            df2 = pd.DataFrame(iris_df.dtypes).transpose()
            df2.index = ['구분']
            st.dataframe(df2)

        with st.expander('기술 통계량'):
            st.dataframe(pd.DataFrame(iris_df.describe()).transpose())

        st.write('타겟분포')
        st.dataframe(iris_df['species'].value_counts())

    elif submenu == '그래프':
        st.subheader('그래프')

        # with st.expander('산점도'):

        #     # plotly 그래프
        #     fig1 = px.scatter(iris_df,
        #                         x = 'sepal_width',
        #                         y = 'sepal_length',
        #                         color = 'species',
        #                         size = 'petal_width',
        #                         hover_data=['petal_length'],
        #                         title='IRIS 산점도')
        #     st.plotly_chart(fig1)

        # layouts
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.subheader('col1')
        #     # seaborn 그래프
        #     fig, ax = plt.subplots()
        #     sns.boxplot(iris_df, x = 'species', y = 'sepal_length', ax=ax)
        #     st.pyplot(fig)

        # with col2:
        #     st.subheader('col2')
        #     # 히스토그램 (Matplotlib)
        #     fig, ax = plt.subplots()
        #     ax.hist(iris_df['sepal_length'], color='green')
        #     st.pyplot(fig)
        

        # Tabs
        tab1, tab2, tab3= st.tabs(['히스토그램', '산점도','박스플롯'])
        val_species = st.selectbox('종 선택', ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))
        st.write('종 선택:', val_species)

        result=iris_df[iris_df['species'] == val_species]
        with tab1:
            st.write('붓꽃의 세 종류인 Setosa, versicolor, virginic 의 꽃받침 길이(Sepal Length), 꽃받침 폭(Sepal Width), 꽃잎 길이(Petal Length),꽃잎 폭(Petal Width)을 히스토그램으로 나타낸 것입니다.')
                  
            fig1, ax = plt.subplots()
            ax.hist(result['sepal_length'],
                        color='red', alpha=0.7)
            ax.hist(result['sepal_width'], 
                        color='yellow', alpha=0.7)
            ax.hist(result['petal_length'], 
                        color='green', alpha=0.7)
            ax.hist(result['petal_width'], 
                        color='blue', alpha=0.7)
            ax.legend(['sepal_length','sepal width','petal length','petal width'])
            st.pyplot(fig1)

            
            
            pie_values = iris_df['species'].value_counts().values.tolist()
            #그래프 생성
            fig2, ax = plt.subplots()
            fig2 = px.scatter(result, 
                                x = 'sepal_width', 
                                y = 'sepal_length', 
                                size='sepal_width',
                               
                                hover_data=['petal_length'])
            st.plotly_chart(fig2)

          
     
        
        with tab2:
            st.write('붓꽃의 세 종류인 Setosa, versicolor, virginic 의 꽃받침 길이(Sepal Length), 꽃받침 폭(Sepal Width)을 산점도로 나타낸 것입니다.')
            fig2, ax = plt.subplots()
            fig2 = px.scatter(result, 
                                x = 'sepal_width', 
                                y = 'sepal_length', 
                                size='sepal_width',
                               
                                hover_data=['petal_length'])
            st.plotly_chart(fig2)

            

        with tab3:
            st.write('붓꽃의 세 종류인 Setosa, versicolor, virginic 의 꽃받침 길이(Sepal Length), 종(species)을 박스플롯으로 나타낸 것입니다.')

            fig3, ax = plt.subplots()
            sns.boxplot(result, 
                            x = 'species', 
                            y = 'sepal_length', 
                            ax=ax)
            st.pyplot(fig3)
           
        with st.expander('히스토그램 종합'):

            fig1, ax = plt.subplots()
            ax.hist(iris_df['sepal_length'],
                        color='red', alpha=0.6)
            ax.hist(iris_df['sepal_width'], 
                        color='yellow', alpha=0.6)
            ax.hist(iris_df['petal_length'], 
                        color='green', alpha=0.6)
            ax.hist(iris_df['petal_width'], 
                        color='blue', alpha=0.6)
            ax.legend(['sepal_length','sepal width','petal length','petal width'])
            st.pyplot(fig1)

        with st.expander('산점도 종합'):

           
            fig2 = px.scatter(iris_df,
                                x = 'sepal_width',
                                y = 'sepal_length',
                                color = 'species',
                                size = 'petal_width',
                                hover_data=['petal_length'],
                                title='IRIS 산점도')
            st.plotly_chart(fig2)

        with st.expander('박스플롯 종합'):

            
            fig, ax = plt.subplots()
            sns.boxplot(iris_df, x = 'species', y = 'sepal_length', ax=ax)
            st.pyplot(fig)


        
    else:
        pass