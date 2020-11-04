NAME = dslr

DSLR = dslr/data.py dslr/math.py LinearRegression.py

NOTEBOOKS = histogram.ipynb  scatter_plot.ipynb pair_plot.ipynb logreg_train.ipynb

$(NAME):

push:
	git add Makefile describe.py $(DSLR) $(NOTEBOOKS)
	git commit -m "dslr"
	git push origin main