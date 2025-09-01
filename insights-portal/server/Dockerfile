# Minimal Dockerfile for pdf-service.js
# Comments in English only.

FROM node:18-bullseye
WORKDIR /app
COPY pdf-service.js ./
# Create a minimal package.json to enable ESM for Node
RUN echo '{"type":"module","dependencies":{"puppeteer":"^22.12.1"}}' > package.json \
 && npm install --omit=dev \
 && apt-get update && apt-get install -y --no-install-recommends \
	 fonts-liberation fonts-noto-color-emoji fonts-ipafont-gothic \
	 libx11-6 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libxkbcommon0 libxshmfence1 \
	 libasound2 libatk1.0-0 libatk-bridge2.0-0 libcairo2 libcups2 libdbus-1-3 libdrm2 libgbm1 libglib2.0-0 libnspr4 libnss3 libpango-1.0-0 \
 && rm -rf /var/lib/apt/lists/*
EXPOSE 8787
CMD ["node", "pdf-service.js"]
