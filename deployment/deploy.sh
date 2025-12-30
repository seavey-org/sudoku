#!/bin/bash

# Configuration
SERVER="192.168.86.227"
APP_DIR="/opt/sudoku"
SERVICE_NAME="sudoku"
DOMAIN="sudoku.seavey.dev"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Deployment to $SERVER...${NC}"

# 1. Build Backend
echo -e "${GREEN}[1/6] Building Backend (Go)...${NC}"
cd backend
GOOS=linux GOARCH=amd64 go build -o sudoku-server .
if [ $? -ne 0 ]; then
    echo "Backend build failed."
    exit 1
fi
cd ..

# 2. Build Frontend
echo -e "${GREEN}[2/6] Building Frontend (Vue.js)...${NC}"
cd frontend
npm install
npm run build
if [ $? -ne 0 ]; then
    echo "Frontend build failed."
    exit 1
fi
cd ..

# 3. Prepare Remote Directory
echo -e "${GREEN}[3/6] Preparing Remote Directory ($APP_DIR)...${NC}"
ssh $SERVER "sudo systemctl stop $SERVICE_NAME || true" # Stop service if running to avoid text file busy
ssh $SERVER "sudo mkdir -p $APP_DIR && sudo chown \$(whoami) $APP_DIR"

# 4. Transfer Files
echo -e "${GREEN}[4/6] Transferring Files...${NC}"
scp backend/sudoku-server $SERVER:$APP_DIR/
scp -r frontend/dist $SERVER:$APP_DIR/

# 5. Setup Systemd Service
echo -e "${GREEN}[5/6] Configuring Systemd Service...${NC}"
scp deployment/sudoku.service $SERVER:/tmp/
ssh $SERVER "sudo mv /tmp/sudoku.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable $SERVICE_NAME && sudo systemctl start $SERVICE_NAME"

# 6. Setup Nginx
echo -e "${GREEN}[6/6] Configuring Nginx...${NC}"
scp deployment/sudoku.seavey.dev.conf $SERVER:/tmp/
ssh $SERVER "sudo mv /tmp/sudoku.seavey.dev.conf /etc/nginx/sites-available/$DOMAIN.conf && sudo ln -sf /etc/nginx/sites-available/$DOMAIN.conf /etc/nginx/sites-enabled/ && sudo nginx -t && sudo systemctl reload nginx"

echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}App should be live at https://$DOMAIN (ensure DNS is pointed to $SERVER)${NC}"
