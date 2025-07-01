import random
import pandas as pd
import streamlit as st
st.title("Game: Word Guessing Game")
st.write("Game Rule: 1. Try to guess the word in limited trials(10)")
st.write("           2. Enter one letter each time")
st.write("           3. If the letter is in the word, it will be displayed")
st.write("           4. If the letter is not in the word, you will lose a life")
st.write("Good luck!")

words = pd.read_csv('3000(new).csv')
words = words[['Word', 'Paraphrase']]
words = words.dropna(subset=['Word', 'Paraphrase'])
word_list = {word: paraphrase for word, paraphrase in zip(words['Word'], words['Paraphrase'])}

while True:
    life = 10
    word = random.choice(list(word_list.keys()))
    paraphrase = word_list.get(word)[0]
    length = len(word)
    guess = ['_'] * length
    st.write(guess)

    while '_' in guess:
        letter = st.text_input('Guess a letter: ')
        if not letter.isalpha():
            st.warning('Please enter a letter!')
            continue
        if letter in word:
            for i, c in enumerate(word):
                if c == letter:
                    guess[i] = letter
            st.write(guess)
        else:
            st.write('The letter is not in the word.')
            life -= 1
            st.write(f'Life - 1. Life: {life}')
            st.write(guess)
        if life == 0:
            st.error('Game over!')
            st.write(f'The answer: {word}')
            st.write(f'Paraphrase: {paraphrase}')
            break
    else:
        st.success('Congratulations, you win!')
        st.write(f'The answer: {word}')
        st.write(f'Paraphrase: {paraphrase}')

    if st.button("Continue"):
        continue
    if st.button("Quit"):
        break