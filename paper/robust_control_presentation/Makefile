# Build pipeline for the robust-control presentation slide deck.
#
#   presentation.md  (symlink to the Obsidian vault source)
#        |   preprocess.py  (strip Obsidian-isms, normalize headings,
#        |                   rewrite Pasted-image filenames)
#        v
#   .presentation_content.md   (intermediate, processed markdown)
#        |   pandoc -t beamer --slide-level=2
#        v
#   presentation_content.tex   (LaTeX frame fragments, \input{} by shell .tex)
#        |   latexmk
#        v
#   presentation.pdf
#
# Each `## heading` and each `---` horizontal rule in the source becomes a
# new slide. The preprocess promotes any stray `# heading` to `## heading`
# so the slide-level rule applies uniformly.

OUTPUT_DIR := /home/jaden/Documents/generated_pdfs
SOURCE     := presentation.md
PROCESSED  := .presentation_content.md
BODY       := presentation_content.tex
SHELL_TEX  := presentation.tex
PDF        := presentation.pdf

PANDOC_FLAGS := \
	--from=markdown+tex_math_dollars+raw_tex-simple_tables-multiline_tables \
	--to=beamer \
	--slide-level=2 \
	--wrap=preserve

# Note on the disabled extensions: pandoc's simple_tables and multiline_tables
# both treat a line of '---' as a table rule. Combined with our slide-break
# convention (a horizontal-rule '---' on its own line), they end up parsing
# the next slide's heading and body as one big single-column table inside the
# previous slide's frame. Disabling them lets '---' be a thematic break, which
# is what pandoc beamer needs for slide separation.

all: $(PDF)
	@mkdir -p $(OUTPUT_DIR)
	@cp $(PDF) $(OUTPUT_DIR)/
	@echo "PDF copied to $(OUTPUT_DIR)/$(PDF)"

# Intermediate: apply Obsidian-isms preprocessor.
$(PROCESSED): $(SOURCE) preprocess.py
	python3 preprocess.py < $(SOURCE) > $(PROCESSED)

# Intermediate: pandoc emits beamer frame fragments (no preamble), since
# the shell .tex provides the documentclass, theme, title, and \maketitle.
$(BODY): $(PROCESSED)
	pandoc $(PANDOC_FLAGS) $(PROCESSED) -o $(BODY)

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
