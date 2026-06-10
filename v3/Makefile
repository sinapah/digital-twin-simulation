PYTHON ?= $(shell if [ -x venv/bin/python ]; then echo venv/bin/python; elif [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
TARGET_HOST ?= 127.0.0.1
PORT ?= 5005
BASE_PATH ?= ../DETRAC-Images/DETRAC-Images
FPS ?= 25
SEQUENCE_COUNT ?= 20
SENDER0_SEQUENCE_START ?= 0
SENDER1_SEQUENCE_START ?= 20
SENDER2_SEQUENCE_START ?= 40
SENDER0_OUTAGE_WINDOWS ?= 20:35,55:70
SENDER1_OUTAGE_WINDOWS ?=
SENDER2_OUTAGE_WINDOWS ?=
FALLBACK_MODE ?= none
OUTAGE_TIMEOUT ?= 2.0
ROUNDS ?= 100
SAMPLES_PER_SENDER_PER_ROUND ?= 32
ROUND_COLLECT_SECONDS ?= 2.0
TRAIN_BATCHES_PER_ROUND ?= 4
REPLAY_BUFFER_SIZE ?= 10000
MAX_CLASS_WEIGHT ?= 5.0
FALLBACK_VIDEO_COUNT ?= 20
LIVE_RESERVED_VIDEO_COUNT ?= 60
TIME_SCALE ?= 1.0

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
	@echo "  sender ranges        camera0=0..19, camera1=20..39, camera2=40..59 by default"
	@echo "  sender outages       SENDER0_OUTAGE_WINDOWS=5:20 SENDER1_OUTAGE_WINDOWS=30:45"

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
		--samples-per-sender-per-round $(SAMPLES_PER_SENDER_PER_ROUND) \
		--round-collect-seconds $(ROUND_COLLECT_SECONDS) \
		--train-batches-per-round $(TRAIN_BATCHES_PER_ROUND) \
		--replay-buffer-size $(REPLAY_BUFFER_SIZE) \
		--max-class-weight $(MAX_CLASS_WEIGHT) \
		--fallback-video-count $(FALLBACK_VIDEO_COUNT) \
		--live-reserved-video-count $(LIVE_RESERVED_VIDEO_COUNT) \
		--time-scale $(TIME_SCALE)

senders: sender0 sender1 sender2

sender0: setup
	@nohup $(PYTHON) sender.py \
		--sender-id camera0 \
		--target-host $(TARGET_HOST) \
		--port $(PORT) \
		--base-path $(BASE_PATH) \
		--sequence-start $(SENDER0_SEQUENCE_START) \
		--sequence-count $(SEQUENCE_COUNT) \
		--fps $(FPS) \
		--outage-windows "$(SENDER0_OUTAGE_WINDOWS)" \
		> $(LOG_DIR)/sender0.log 2>&1 & echo $$! > $(RUNTIME_DIR)/sender0.pid
	@echo "sender0 PID $$(cat $(RUNTIME_DIR)/sender0.pid)"

sender1: setup
	@nohup $(PYTHON) sender.py \
		--sender-id camera1 \
		--target-host $(TARGET_HOST) \
		--port $(PORT) \
		--base-path $(BASE_PATH) \
		--sequence-start $(SENDER1_SEQUENCE_START) \
		--sequence-count $(SEQUENCE_COUNT) \
		--fps $(FPS) \
		--outage-windows "$(SENDER1_OUTAGE_WINDOWS)" \
		> $(LOG_DIR)/sender1.log 2>&1 & echo $$! > $(RUNTIME_DIR)/sender1.pid
	@echo "sender1 PID $$(cat $(RUNTIME_DIR)/sender1.pid)"

sender2: setup
	@nohup $(PYTHON) sender.py \
		--sender-id camera2 \
		--target-host $(TARGET_HOST) \
		--port $(PORT) \
		--base-path $(BASE_PATH) \
		--sequence-start $(SENDER2_SEQUENCE_START) \
		--sequence-count $(SEQUENCE_COUNT) \
		--fps $(FPS) \
		--outage-windows "$(SENDER2_OUTAGE_WINDOWS)" \
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
	@$(MAKE) senders FPS=5 SEQUENCE_COUNT=1 SENDER0_OUTAGE_WINDOWS=1:2
	@sleep 5
	@$(MAKE) stop
