########################################################################################################################
# Imports
########################################################################################################################
import cbrcmndr
import requests

import streamlit as st
import pandas as pd



########################################################################################################################
# Functions
########################################################################################################################
def fetch_mov_data(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={st.secrets["TMDB_API_KEY"]}&language=en-US')
    data = response.json()
    #print(data)
    title = data['title']
    poster_path = 'https://image.tmdb.org/t/p/w500/' + data['poster_path']
    return title, poster_path


########################################################################################################################
# Main
########################################################################################################################
if __name__ == '__main__':
    st.set_page_config(page_title='CB Movie Recommender', page_icon=':robot_face:')
    
    #st.title('Content Based Movie Recommendation System Using Vector Cosine Similarity')
    st.markdown("""<style>.big-font {font-size:32px !important; font-weight: bold;}</style>""", unsafe_allow_html=True)
    st.markdown('<p class="big-font">Content Based Movie Recommendation System Using Vector Cosine Similarity</p>', unsafe_allow_html=True)
    st.caption("Made by [Dibyajyoti Sarkar](https://www.linkedin.com/in/djsarkar93)")
    st.divider()
    
    sel_mov_title = st.selectbox('Please choose a movie:', cbrcmndr.movies_tags_df['title'].values)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:    
        rcmnd_btn = st.button('Find Similar Movies')
    st.divider()
    
    if rcmnd_btn:
        rcmndd_ids = cbrcmndr.recommend(movie_title = sel_mov_title)
        
        recommendations = []
        for mid in rcmndd_ids:
            recommendations.append( (mid, *fetch_mov_data(mid)) )
        
        col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
        with col1:
            st.image( recommendations[0][2] )
            st.markdown( recommendations[0][1] )
        with col2:
            st.image( recommendations[1][2] )
            st.markdown( recommendations[1][1] )
        with col3:
            st.image( recommendations[2][2] )
            st.markdown( recommendations[2][1] )
        with col4:
            st.image( recommendations[3][2] )
            st.markdown( recommendations[3][1] )
        with col5:
            st.image( recommendations[4][2] )
            st.markdown( recommendations[4][1] )
    