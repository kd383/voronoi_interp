# compile latex files

.PHONY : all clean
SRC=$(wildcard *.tex)
TAR=$(patsubst %.tex, %.pdf, $(SRC))
all : $(TAR)
%.pdf : %.tex
	pdflatex $*

clean:
	rm -f *.log *.aux *.gz *.out *.fdb_latexmk *.fls

deep_clean: clean
	rm *.pdf
