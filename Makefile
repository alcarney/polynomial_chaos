
default: dissertation

dissertation:
	pdflatex main.tex -o mmath_dissertation.pdf

clean:
	rm *.pdf *.aux *.log
