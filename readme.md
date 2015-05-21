### This is the source of my blog entries. 

If you want to see the blog itself, go to [https://arokem.github.io](https://arokem.github.io)

The content of the blog is written using the [IPython](http://ipython) notebook (all stored in the `content` directory and its sub-directories for `data` and `images` where those are needed), and the pages are generated using (Pelican)[getpelican.com]. Pelican is a Python-based static web-page generator, that has many [themes](https://github.com/getpelican/pelican-themes) to choose from and many [plugins](https://github.com/getpelican/pelican-plugins), to control the flexible generation of web-pages. In particular, to render the notebook into web pages, I also use the [pelican-ipynb](https://github.com/danielfrg/pelican-ipynb) plugin.


### Publishing the blog
Considering that it invloves some delicate operations, the following instructions should be followed with some care and should not be done when tired or drunk. 

To deploy the blog to the web execute the following steps, in order. Don't hesitate to throw a `git status` and an `ls` or `pwd` in there, to see that you are not screwing something up:

	cd output
	git init
	touch .nojekyll
	git add *
	git remote add origin https://github.com/arokem/arokem.github.io
	git commit -a -m"Publish"
	git push -f origin master
	rm -rf .git
	cd ..
