NAME = dslr

DSLR = dslr/data.py dslr/math.py LogisticRegression.py houses.csv

NOTEBOOKS =	histogram.ipynb  scatter_plot.ipynb pair_plot.ipynb \
			logreg_train.ipynb logreg_predict.ipynb

launch:
	jupyter notebook

push:
	git add Makefile describe.py $(DSLR) $(NOTEBOOKS)
	git commit -m "final version"
	git push origin main