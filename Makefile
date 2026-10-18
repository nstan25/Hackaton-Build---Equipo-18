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
	@test "$(SOURCE)" = sdb_bcs || { echo "preprocess solo aplica a SOURCE=sdb_bcs" >&2; exit 2; }
	$(RUN) python -m bcs.data.preprocess_sdb $(if $(SAMPLE),--sample $(SAMPLE))

data:
	@test -n "$(SOURCE)" || { echo "uso: make data SOURCE=<source_id>" >&2; exit 2; }
	$(RUN) python scripts/data/fetch_$(SOURCE).py $(if $(SAMPLE),--sample $(SAMPLE))
	if [ "$(SOURCE)" = sdb_bcs ]; then $(MAKE) preprocess SOURCE=$(SOURCE) $(if $(SAMPLE),SAMPLE=$(SAMPLE)); fi
	$(RUN) python -m bcs.data.sources.$(SOURCE) $(if $(SAMPLE),--sample $(SAMPLE))

data-sample:
	@test -n "$(SOURCE)" || { echo "uso: make data-sample SOURCE=<source_id>" >&2; exit 2; }
	$(MAKE) data SOURCE=$(SOURCE) SAMPLE=60

labels:
	$(RUN) python -m bcs.data.merge

# G1: make eval usa la grilla del modelo (dairy_sdb) salvo que se pise BCS_SCALE.
eval: export BCS_SCALE ?= dairy_sdb
eval:
	@test -n "$(MODEL)" -a -n "$(STAGE)" || { echo "uso: make eval MODEL=<name> STAGE=<m2|m3|m4>" >&2; exit 2; }
	$(RUN) python eval/evaluate.py --model $(MODEL) --stage $(STAGE) $(if $(PREDICTIONS),--predictions $(PREDICTIONS)) $(if $(QUALITY),--quality $(QUALITY)) $(if $(TASK),--task $(TASK))

pack-data:
	$(RUN) python scripts/compute/pack_data.py

train:
	@test -n "$(BACKEND)" || { echo "uso: make train [MODEL=<m>] BACKEND=<modal|kaggle|tenki|local> [SMOKE=1] [FOLDS=all] [TASK=<id>]" >&2; exit 2; }
	$(RUN) python scripts/compute/launch.py --model $(or $(MODEL),stub_uniform) --backend $(BACKEND) $(if $(FOLDS),--folds $(FOLDS)) $(if $(SMOKE),--smoke) $(if $(TASK),--task $(TASK))

browser:
	$(RUN) python tests/browser/run_browser.py

# Solo el humano (SPEC §4.1). Los agentes tienen estos targets en deny.
holdout:
	$(call pending,el humano en GH)

eval-final:
	$(call pending,el humano en G4)

deploy:
	$(call pending,T01 con permiso del humano)
