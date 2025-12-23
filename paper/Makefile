# Output directory for generated PDFs
OUTPUT_DIR := /home/jaden/Documents/generated_pdfs

# Default target: build both versions and copy them
all: ieee_ethereal_weights.pdf ieee_ethereal_weights_appendixed.pdf
	@echo "Copying PDFs to $(OUTPUT_DIR)..."
	@mkdir -p $(OUTPUT_DIR)
	@cp ieee_ethereal_weights.pdf $(OUTPUT_DIR)/
	@cp ieee_ethereal_weights_appendixed.pdf $(OUTPUT_DIR)/
	@echo "✓ PDFs copied to $(OUTPUT_DIR)"

# Build main paper (no appendix)
ieee_ethereal_weights.pdf: ieee_ethereal_weights.tex paper_content.tex
	latexmk -pdf ieee_ethereal_weights.tex

# Build paper with appendix
ieee_ethereal_weights_appendixed.pdf: ieee_ethereal_weights_appendixed.tex paper_content.tex appendix.tex
	latexmk -pdf ieee_ethereal_weights_appendixed.tex

# Watch targets
watch-no-appendix:
	latexmk -pvc -pdf ieee_ethereal_weights.tex

watch-appendix:
	latexmk -pvc -pdf ieee_ethereal_weights_appendixed.tex

# Clean build artifacts
clean:
	latexmk -C
	rm -rf build tikz-cache _minted-*

.PHONY: all watch-no-appendix watch-appendix clean
