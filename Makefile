NAME = dslr

$(NAME):

push:
	git add Makefile describe.py dslr/*.py
	git commit -m "dslr"
	git push origin main