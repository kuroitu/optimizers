python = python3.8
srcs = $(wildcard ./_*.py)

test:
	@for src in $(srcs); do \
		$(python) $$src; \
	done
