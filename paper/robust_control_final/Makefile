# Build pipeline for the robust-control paper.
#
#   report_skeleton.md  (symlink to the Obsidian vault draft)
#        |   preprocess.py  (strip Obsidian-isms, normalize headings)
#        v
#   .paper_content.md   (intermediate, processed markdown)
#        |   pandoc
#        v
#   paper_content.tex   (LaTeX body, \input{} by the shell .tex)
#        |   latexmk
#        v
#   robust_control.pdf

OUTPUT_DIR := /home/jaden/Documents/generated_pdfs
SKELETON   := report_skeleton.md
PROCESSED  := .paper_content.md
BODY       := paper_content.tex
SHELL_TEX  := robust_control.tex
PDF        := robust_control.pdf

PANDOC_FLAGS := \
	--from=markdown+tex_math_dollars+raw_tex \
	--to=latex \
	--wrap=preserve

all: $(PDF)
	@mkdir -p $(OUTPUT_DIR)
	@cp $(PDF) $(OUTPUT_DIR)/
	@echo "PDF copied to $(OUTPUT_DIR)/$(PDF)"

# Intermediate: apply Obsidian-isms preprocessor.
$(PROCESSED): $(SKELETON) preprocess.py
	python3 preprocess.py < $(SKELETON) > $(PROCESSED)

# Intermediate: pandoc into the body fragment, then postprocess to fix
# constructs that don't work with the two-column ieeeconf class (longtable
# etc.). Pandoc → stdout → postprocess.py → file.
$(BODY): $(PROCESSED) postprocess.py
	pandoc $(PANDOC_FLAGS) $(PROCESSED) | python3 postprocess.py > $(BODY)

# Final PDF: compile the shell, which \input{}s the generated body.
$(PDF): $(SHELL_TEX) $(BODY)
	latexmk -pdf $(SHELL_TEX)

# Rebuild on every save of the source markdown.
watch:
	latexmk -pvc -pdf $(SHELL_TEX)

# Produce the body fragment only, for inspection without compiling.
tex-only: $(BODY)

clean:
	latexmk -C
	rm -f $(PROCESSED) $(BODY)

.PHONY: all watch tex-only clean
