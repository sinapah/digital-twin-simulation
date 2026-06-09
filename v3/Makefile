PYTHON ?= $(shell if [ -x venv/bin/python ]; then echo venv/bin/python; elif [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
TARGET_HOST ?= 127.0.0.1
PORT ?= 5005
BASE_PATH ?= ../DETRAC-Images/DETRAC-Images
FPS ?= 25
SEQUENCE_COUNT ?= 10
OUTAGE_WINDOWS ?= 20:35,55:70
FALLBACK_MODE ?= none
OUTAGE_TIMEOUT ?= 2.0
ROUNDS ?= 100
TRAIN_BATCHES_PER_ROUND ?= 4
REPLAY_BUFFER_SIZE ?= 10000
MAX_CLASS_WEIGHT ?= 5.0

RUNTIME_DIR := runtime
LOG_DIR := logs
OUTPUT_DIR := outputs

.PHONY: help setup receiver train-receiver senders sender0 sender1 sender2 stop clean-logs smoke-senders

help:
	@echo "v3 experiment targets:"
	@echo "  make setup           Create runtime, log, output, and visualization dirs"
	@echo "  make receiver        Start receiver in background"
	@echo "  make train-receiver  Start receiver-side live training in foreground"
	@echo "  make senders         Start three sender processes in background"
	@echo "  make sender0         Start sender camera0 only"
	@echo "  make sender1         Start sender camera1 only"
	@echo "  make sender2         Start sender camera2 only"
	@echo "  make stop            Stop only PIDs recorded by this Makefile"
	@echo "  make clean-logs      Remove v3 runtime logs and PID files"

setup:
	@mkdir -p $(RUNTIME_DIR) $(LOG_DIR) $(OUTPUT_DIR) visualizations

receiver: setup
	@nohup $(PYTHON) receiver.py \
		--port $(PORT) \
		--fallback-mode $(FALLBACK_MODE) \
		--outage-timeout $(OUTAGE_TIMEOUT) \
		> $(LOG_DIR)/receiver.log 2>&1 & echo $$! > $(RUNTIME_DIR)/receiver.pid
	@echo "receiver PID $$(cat $(RUNTIME_DIR)/receiver.pid)"

train-receiver: setup
	$(PYTHON) receiver_training.py \
		--port $(PORT) \
		--fallback-mode $(FALLBACK_MODE) \
		--outage-timeout $(OUTAGE_TIMEOUT) \
		--rounds $(ROUNDS) \
		--train-batches-per-round $(TRAIN_BATCHES_PER_ROUND) \
		--replay-buffer-size $(REPLAY_BUFFER_SIZE) \
		--max-class-weight $(MAX_CLASS_WEIGHT)

senders: sender0 sender1 sender2

sender0: setup
	@nohup $(PYTHON) sender.py \
		--sender-id camera0 \
		--target-host $(TARGET_HOST) \
		--port $(PORT) \
		--base-path $(BASE_PATH) \
		--sequence-start 0 \
		--sequence-count $(SEQUENCE_COUNT) \
		--fps $(FPS) \
		--outage-windows "$(OUTAGE_WINDOWS)" \
		> $(LOG_DIR)/sender0.log 2>&1 & echo $$! > $(RUNTIME_DIR)/sender0.pid
	@echo "sender0 PID $$(cat $(RUNTIME_DIR)/sender0.pid)"

sender1: setup
	@nohup $(PYTHON) sender.py \
		--sender-id camera1 \
		--target-host $(TARGET_HOST) \
		--port $(PORT) \
		--base-path $(BASE_PATH) \
		--sequence-start 10 \
		--sequence-count $(SEQUENCE_COUNT) \
		--fps $(FPS) \
		--outage-windows "" \
		> $(LOG_DIR)/sender1.log 2>&1 & echo $$! > $(RUNTIME_DIR)/sender1.pid
	@echo "sender1 PID $$(cat $(RUNTIME_DIR)/sender1.pid)"

sender2: setup
	@nohup $(PYTHON) sender.py \
		--sender-id camera2 \
		--target-host $(TARGET_HOST) \
		--port $(PORT) \
		--base-path $(BASE_PATH) \
		--sequence-start 20 \
		--sequence-count $(SEQUENCE_COUNT) \
		--fps $(FPS) \
		--outage-windows "" \
		> $(LOG_DIR)/sender2.log 2>&1 & echo $$! > $(RUNTIME_DIR)/sender2.pid
	@echo "sender2 PID $$(cat $(RUNTIME_DIR)/sender2.pid)"

stop:
	@for pidfile in $(RUNTIME_DIR)/*.pid; do \
		if [ -f "$$pidfile" ]; then \
			pid=$$(cat "$$pidfile"); \
			if kill -0 "$$pid" 2>/dev/null; then \
				echo "Stopping $$pid from $$pidfile"; \
				kill "$$pid"; \
			fi; \
			rm -f "$$pidfile"; \
		fi; \
	done

clean-logs:
	@rm -f $(LOG_DIR)/*.log $(RUNTIME_DIR)/*.pid

smoke-senders:
	@$(MAKE) receiver FALLBACK_MODE=fixed OUTAGE_TIMEOUT=0.5
	@sleep 1
	@$(MAKE) senders FPS=5 SEQUENCE_COUNT=1 OUTAGE_WINDOWS=1:2
	@sleep 5
	@$(MAKE) stop
