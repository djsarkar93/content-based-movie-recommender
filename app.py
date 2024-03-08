########################################################################################################################
# Imports
########################################################################################################################
import cbrcmndr

import streamlit as st
import pandas as pd



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
        rcmnd_btn = st.button('See Similar Movies')
    st.divider()
    
    if rcmnd_btn:
        rcmndd_ids = cbrcmndr.recommend(movie_title = sel_mov_title)
        st.write(rcmndd_ids)
    