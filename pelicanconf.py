#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Ariel Rokem'
SITENAME = 'Ariel Rokem'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Europe/Paris'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Website', 'http://arokem.org'))

# Social widget
SOCIAL = (('Twitter', 'https://twitter.com/arokem'),
          ('Github', 'https://github.com/arokem'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

THEME = 'pelican-elegant/'

PLUGINS = ['pelican-plugins.pelican-ipynb']
MARKUP = ('md', 'ipynb')
