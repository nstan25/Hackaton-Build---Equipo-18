# Targets obligatorios de SPEC §3.2. Los marcados "pendiente" los implementa la tarea indicada.
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
RUN := $(UV) run --frozen
SOURCE ?=
MODEL ?=
STAGE ?=
BACKEND ?= local

define pending
	@echo "make $@: pendiente, lo implementa $(1)" >&2; exit 1
endef

.PHONY: check smoke preprocess data data-sample labels eval pack-data train browser \
	holdout eval-final deploy fixtures

check:
	$(RUN) ruff format --check .
	$(RUN) ruff check .
	$(RUN) pyright
	$(RUN) pytest -q

smoke:
	$(RUN) python scripts/smoke.py

fixtures:
	$(RUN) python scripts/make_fixtures.py

preprocess:
	@test -n "$(SOURCE)" || { echo "uso: make preprocess SOURCE=sdb_bcs" >&2; exit 2; }
	$(call pending,T03)

data:
	@test -n "$(SOURCE)" || { echo "uso: make data SOURCE=<source_id>" >&2; exit 2; }
	$(call pending,T03/T04/T05)

data-sample:
	@test -n "$(SOURCE)" || { echo "uso: make data-sample SOURCE=<source_id>" >&2; exit 2; }
	$(call pending,T03/T04/T05)

labels:
	$(call pending,T06)

eval:
	@test -n "$(MODEL)" -a -n "$(STAGE)" || { echo "uso: make eval MODEL=<name> STAGE=<m2|m3|m4>" >&2; exit 2; }
	$(RUN) python eval/evaluate.py --model $(MODEL) --stage $(STAGE)

pack-data:
	$(call pending,T16)

train:
	@test -n "$(MODEL)" || { echo "uso: make train MODEL=<m> BACKEND=<modal|kaggle|tenki|local> [SMOKE=1]" >&2; exit 2; }
	$(call pending,T16)

browser:
	$(call pending,T07)

# Solo el humano (SPEC §4.1). Los agentes tienen estos targets en deny.
holdout:
	$(call pending,el humano en GH)

eval-final:
	$(call pending,el humano en G4)

deploy:
	$(call pending,T01 con permiso del humano)
