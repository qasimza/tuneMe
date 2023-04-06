# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, DateField, SelectField, DecimalField, BooleanField
from wtforms.validators import Email, DataRequired, NumberRange

# login, registration and search fields


class LoginForm(FlaskForm):
    username = StringField('Username',
                         id='username_login',
                         validators=[DataRequired()])
    password = PasswordField('Password',
                             id='pwd_login',
                             validators=[DataRequired()])


class CreateAccountForm(FlaskForm):
    username = StringField('Username',
                         id='username_create',
                         validators=[DataRequired()])
    email = StringField('Email',
                      id='email_create',
                      validators=[DataRequired(), Email()])
    password = PasswordField('Password',
                             id='pwd_create',
                             validators=[DataRequired()])
    
class PerformSearchFrom(FlaskForm):
    
    # Prelimnary Search Options
    song_title = StringField('Songtitle', id='song_title')
    artist = StringField('Artist', id='artist')
    year = DateField('Year', id='year', format='%Y')
    genres = SelectField('Genre', id='genres')
    themes = SelectField('Themes', id='themes')
    
    # Advanced Search Options
    popularity = DecimalField('Popularity', id='popularity')
    danceability = DecimalField('Danceability', id='danceability')
    energy = DecimalField('Energy', id='energy')
    loudness = DecimalField('Loudness', id='loudness')
    speechiness = DecimalField('Speechiness', id='speechiness')
    accoustics = DecimalField('Accoustics', id='accoustics')
    instrumentalness = DecimalField('Instrumentalness', id='instrumentalness')
    liveliness = DecimalField('Liveliness', id='liveliness')
    valence = DecimalField('Valence', id='valence')
    tempo = DecimalField('Tempo', id='tempo')
    explicit = BooleanField('Explicit', id='explicit')



   