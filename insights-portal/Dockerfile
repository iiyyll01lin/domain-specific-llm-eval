# =============================================================================
# Stage 1 — Builder
# Install dependencies and compile the Vite/React application.
# =============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Copy manifests first for layer-cache efficiency
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# =============================================================================
# Stage 2 — Production
# Serve the compiled static assets with Nginx.
# =============================================================================
FROM nginx:alpine AS production

# Remove the default Nginx welcome page
RUN rm -rf /usr/share/nginx/html/*

# SPA routing: redirect all 404s back to index.html so React Router works
COPY <<'EOF' /etc/nginx/conf.d/default.conf
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Serve pre-compressed assets when available
    gzip_static on;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache hashed static assets for a long time
    location ~* \.(js|css|woff2?|ttf|eot|ico|svg|png|jpg|jpeg|gif|webp)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Copy compiled assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
