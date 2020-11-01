NAME = dslr

DSLR = dslr/data.py dslr/math.py

$(NAME):

push:
	git add Makefile describe.py $(DSLR)
	git commit -m "dslr"
	git push origin main