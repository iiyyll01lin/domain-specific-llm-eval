# Minimal Dockerfile for pdf-service.js
# Comments in English only.

FROM node:18-alpine
WORKDIR /app
COPY pdf-service.js ./
# Create a minimal package.json to enable ESM for Node
RUN echo '{"type":"module"}' > package.json
EXPOSE 8787
CMD ["node", "pdf-service.js"]
