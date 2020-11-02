NAME = dslr

DSLR = dslr/data.py dslr/math.py

NOTEBOOKS = histogram.ipynb  scatter_plot.ipynb pair_plot.ipynb

$(NAME):

push:
	git add Makefile describe.py $(DSLR) $(NOTEBOOKS)
	git commit -m "dslr"
	git push origin main