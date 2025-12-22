# Build both versions of the paper
all: ieee_ethereal_weights.pdf ieee_ethereal_weights_appendixed.pdf

ieee_ethereal_weights.pdf: ieee_ethereal_weights.tex paper_content.tex
	latexmk -pdf ieee_ethereal_weights.tex

ieee_ethereal_weights_appendixed.pdf: ieee_ethereal_weights_appendixed.tex paper_content.tex appendix.tex
	latexmk -pdf ieee_ethereal_weights_appendixed.tex

watch-no-appendix:
	latexmk -pvc -pdf ieee_ethereal_weights.tex

watch-appendix:
	latexmk -pvc -pdf ieee_ethereal_weights_appendixed.tex

clean:
	latexmk -C
	rm -rf build tikz-cache _minted-*

.PHONY: all watch-no-appendix watch-appendix clean