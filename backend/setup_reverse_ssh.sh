#!/bin/bash
# Reverse SSH Tunnel Setup Script
# Run this on the GPU machine (backend) to create a reverse SSH tunnel to the public server

# Configuration - UPDATE THESE VALUES
PUBLIC_SERVER_USER="your-username"
PUBLIC_SERVER_HOST="your-public-server.com"
PUBLIC_SERVER_PORT=22
BACKEND_PORT=8000
TUNNEL_PORT=8000  # Port on public server that will forward to backend

echo "Setting up reverse SSH tunnel..."
echo "Backend (GPU machine) -> Public Server"
echo "Local port $BACKEND_PORT will be accessible on public server at localhost:$TUNNEL_PORT"
echo ""

# Create reverse SSH tunnel
# -R: Reverse tunnel (public_server:port -> local:port)
# -N: Don't execute remote commands
# -f: Background
# -o ExitOnForwardFailure=yes: Exit if port forwarding fails
# -o ServerAliveInterval=60: Keep connection alive
ssh -R ${TUNNEL_PORT}:localhost:${BACKEND_PORT} \
    -N \
    -f \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    ${PUBLIC_SERVER_USER}@${PUBLIC_SERVER_HOST} -p ${PUBLIC_SERVER_PORT}

if [ $? -eq 0 ]; then
    echo "✓ Reverse SSH tunnel established successfully!"
    echo "Backend API is now accessible on public server at localhost:$TUNNEL_PORT"
    echo ""
    echo "To check if tunnel is running:"
    echo "  ssh $PUBLIC_SERVER_USER@$PUBLIC_SERVER_HOST 'netstat -tlnp | grep $TUNNEL_PORT'"
    echo ""
    echo "To kill the tunnel:"
    echo "  ssh $PUBLIC_SERVER_USER@$PUBLIC_SERVER_HOST 'pkill -f \"ssh.*-R.*$TUNNEL_PORT\"'"
    echo "  Or: ps aux | grep 'ssh.*-R.*$TUNNEL_PORT' | grep -v grep | awk '{print \$2}' | xargs kill"
else
    echo "✗ Failed to establish reverse SSH tunnel"
    echo "Please check:"
    echo "  1. SSH key authentication is set up"
    echo "  2. Public server allows remote port forwarding (GatewayPorts in sshd_config)"
    echo "  3. Port $TUNNEL_PORT is not already in use on public server"
    exit 1
fi

