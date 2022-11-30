# -*- coding: utf-8 -*-

# core pkgs
import streamlit as st

# data pkgs
import pandas as pd

# 이미지 라이브러리
from PIL import Image

def main():
    """코드작성"""
    # st.title("Hello World!")
    # st.subheader("임시...")

    # # text
    # num = 1
    # st.text(f'숫자는 {num}')

    # # 마크다운
    # st.markdown('## 마크다운 제목 2번째')

    # # 색상
    # st.success("성공")
    # st.warning("경고")
    # st.info("정보")
    # st.error("에러")
    # st.exception("예외")

    # # superfunction
    # st.write('문자열')
    # st.write(1+2)
    # st.write(dir(int))


# ----------------------------------------------------

    # iris_df = pd.read_csv('data/iris.csv')
    # st.dataframe(iris_df)
    # st.dataframe(iris_df, width=1000, height=1000)

    # # 색상 추가
    # st.title('데이터 프레임에 색상 입히기')
    # st.dataframe(iris_df.style.highlight_max(axis=0))

    # # 코드 보여주기
    # myCode = """
    # def hello():
    # print("Hello World")
    # end
    # """
    # st.code(myCode, language='python')


# ----------------------------------------------------

    # # 버튼 입력
    # name = 'Ryeom'

    # if st.button("제출"):
    #     st.write(f'이름 : {name.upper()}')

    # if st.button("소문자"):
    #     st.write(f'이름 : {name.lower()}')  

    # # 활성화, 비활성화
    # status = st.radio('Status', ("활성화","비활성화"))
    # if status == "활성화":
    #     st.success("활성화 상태")
    # else:
    #     st.error("비활성화 상태")

    # # 체크박스
    # if st.checkbox("Show/Hide"):
    #     st.text("무언가 보여드리겠습니다.")

    # with st.expander('python'):
    #     st.text("Hello Python")



# ----------------------------------------------------



    # # Select/Slider

    # # 프로그래밍 언어
    # proLan = ['파이썬','자바','R','SQL']
    # choice = st.selectbox('휴먼 프로그래밍 언어', proLan)
    # st.write(f'{choice} 언어를 선택함')

    # mulChoice = st.multiselect("언어 선택", proLan, default="SQL")
    # st.write('선택:', mulChoice)

    # # 슬라이더
    # age=st.slider('나이', 1, 120 )
    # st.write(age)

    # #select Slider
    # color = st.select_slider('색상 선택', 
    #                         options=['노란색','빨간색','파란색','흰색'],
    #                         value=('노란색','흰색'))
    # st.write(color)

    # start_color, end_color = st.select_slider('색상 선택', 
    #                         options=['노란색','빨간색','파란색','흰색','검정색'],
    #                         value=('노란색','검정색'))
    # st.write(start_color, end_color)


# ----------------------------------------------------


    # # 이미지 추가
    # img = Image.open('data/image_02.jpg')
    # st.image(img)

    # # URL 이미지 삽입
    # st.image('data:image/png;base64,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')

    # # 비디오 출력
    # with open('data/secret_of_success.mp4', 'rb') as rb:
    #     video_file=rb.read()
    #     st.video(video_file, start_time=1)

    # # 오디오 출력
    # with open('data/song.mp3', 'rb') as rb:
    #     audio_file=rb.read()
    #     st.audio(audio_file, format='audio/mp3')


# ----------------------------------------------------


    # 텍스트 Input
    fname = st.text_input('이름 입력')
    st.write(fname)

    # 텍스트 영역
    message = st.text_area('입력해주세요', height=100)
    st.write(message)

    # 숫자 입력
    number = st.number_input('숫자 입력')
    st.write(number)

    # 날짜 입력
    myDate = st.date_input('날짜')
    st.write(myDate)

    # 시간
    myTime = st.time_input("시간")
    st.write(myTime)

    # color picker
    color = st.color_picker("색상 선택")
    st.write(color)



if __name__=="__main__":
    main()

