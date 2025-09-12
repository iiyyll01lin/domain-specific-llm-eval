IMAGE_NAME?=rag-eval
VERSION?=$(shell cat VERSION 2>/dev/null || echo 0.1.0)
GIT_SHA:=$(shell git rev-parse --short HEAD)

.PHONY: help build tag all

help:
	@echo "Targets: build, tag, compose, dev, validate-compose"

build:
	docker build -t $(IMAGE_NAME):dev .

tag: build
	bash scripts/tag_image.sh IMAGE_NAME=$(IMAGE_NAME) VERSION=$(VERSION)

compose:
	docker compose -f docker-compose.services.yml up -d

dev:
	docker compose -f docker-compose.services.yml -f docker-compose.dev.override.yml up -d --build

validate-compose:
	python scripts/validate_compose.py
