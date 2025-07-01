import random
import pandas as pd
import streamlit as st
st.title("Game: Word Guessing Game")
st.write("Game Rule: ")
st.markdown("""
1. Try to guess the word in limited trials (10)
2. Enter one letter each time  
3. If the letter is in the word, it will be displayed  
4. If not, you lose a life  
""")

@st.cache_data
def load_words():
    df = pd.read_csv('A2_1000_Vocabulary.csv')
    df = df[['Word', 'Meaning']].dropna()
    return {row['Word']: row['Meaning'] for _, row in df.iterrows()}

word_list = load_words()

if "word" not in st.session_state:
    st.session_state.word = random.choice(list(word_list.keys()))
    st.session_state.meaning = word_list[st.session_state.word]
    st.session_state.guess = ['_'] * len(st.session_state.word)
    st.session_state.life = 10
    st.session_state.letters_guessed = []

st.write("Word:", ' '.join(st.session_state.guess))
st.write(f"Lives remaining: {st.session_state.life}")
st.write(f"Letters guessed: {', '.join(st.session_state.letters_guessed)}")

letter = st.text_input("Guess a letter: ", max_chars=1, key="current_guess")

if st.button("Submit Guess"):
    letter = st.session_state.current_guess.strip()
    st.session_state.current_guess = ""  

    if not letter.isalpha():
        st.warning("Please enter a valid letter.")
    elif letter in st.session_state.letters_guessed:
        st.warning("You've already guessed that letter.")
    else:
        st.session_state.letters_guessed.append(letter)

        if letter in st.session_state.word:
            for i, c in enumerate(st.session_state.word):
                if c == letter:
                    st.session_state.guess[i] = letter
            st.success("Correct guess!")
        else:
            st.session_state.life -= 1
            st.error("Wrong guess!")

if '_' not in st.session_state.guess:
    st.success("🎉 You win!")
    st.balloons()
    st.write(f"The word was: **{st.session_state.word}**")
    st.write(f"Meaning: {st.session_state.meaning}")
    if st.button("Play Again"):
        st.session_state.clear()

elif st.session_state.life == 0:
    st.error("💀 Game over!")
    st.write(f"The word was: **{st.session_state.word}**")
    st.write(f"Meaning: {st.session_state.meaning}")
    if st.button("Try Again"):
        st.session_state.clear()
    
