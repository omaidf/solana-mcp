#!/bin/bash
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Docker build process for Solana MCP...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker is not running. Please start Docker desktop or Docker daemon first.${NC}"
  exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo -e "${YELLOW}Creating sample .env file...${NC}"
  cat > .env << EOL
# Solana RPC settings
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_COMMITMENT=confirmed
SOLANA_TIMEOUT=30

# Server settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
LOG_FORMAT=json
ENVIRONMENT=development

# Cache settings
METADATA_CACHE_SIZE=100
METADATA_CACHE_TTL=300
PRICE_CACHE_SIZE=500
PRICE_CACHE_TTL=60
EOL
  echo -e "${GREEN}.env file created. Please review and modify as needed.${NC}"
fi

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t solana-mcp:latest .

# Check if build was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Docker image built successfully!${NC}"
  echo -e "${YELLOW}You can run the container with:${NC}"
  echo -e "docker run -p 8000:8000 solana-mcp:latest"
  echo -e "${YELLOW}Or use docker-compose:${NC}"
  echo -e "docker-compose up -d"
else
  echo -e "${RED}Docker build failed. Please check the error message above.${NC}"
  exit 1
fi 