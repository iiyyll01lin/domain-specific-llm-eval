IMAGE_NAME?=rag-eval
VERSION?=$(shell cat VERSION 2>/dev/null || echo 0.1.0)
GIT_SHA:=$(shell git rev-parse --short HEAD)

.PHONY: help build tag build-tag compose dev validate-compose compose-prebuilt smoke-prebuilt all

PREBUILT_ENV_FILE?=.env.prebuilt

help:
	@echo "Targets: build, tag, build-tag, compose, dev, validate-compose, compose-prebuilt, smoke-prebuilt"

build:
	docker build -t $(IMAGE_NAME):dev .

tag:
	bash scripts/tag_image.sh IMAGE_NAME=$(IMAGE_NAME) VERSION=$(VERSION)

build-tag: build tag

compose:
	docker compose -f docker-compose.services.yml up -d

dev:
	docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml up -d --build

validate-compose:
	python3 scripts/validate_compose.py

compose-prebuilt:
	COMPOSE_ENV_FILE=$(PREBUILT_ENV_FILE) docker compose --env-file $(PREBUILT_ENV_FILE) -f docker-compose.services.yml up -d --no-build

smoke-prebuilt:
	COMPOSE_ENV_FILE=$(PREBUILT_ENV_FILE) SMOKE_USE_PREBUILT_IMAGE=1 bash scripts/e2e_smoke.sh
