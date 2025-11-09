#!/bin/bash
# Raspberry Pi 3 Setup Script for Home Assistant STT Toolkit

echo "Setting up Home Assistant STT Toolkit for Raspberry Pi 3..."

# Check if running on Raspberry Pi
if [[ $(uname -m) == arm* ]] || [[ $(cat /proc/cpuinfo | grep -i "raspberry pi") ]]; then
    echo "✓ Raspberry Pi detected"
else
    echo "⚠️  Warning: This script is designed for Raspberry Pi"
fi

# Update system packages
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    python3-pyaudio \
    alsa-utils \
    sox \
    ffmpeg \
    git \
    build-essential \
    python3-dev \
    libasound2-dev

# Install audio tools for testing
sudo apt install -y arecord aplay

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Raspberry Pi optimized requirements
echo "Installing Python packages optimized for RPi 3..."
pip install -r requirements-rpi.txt

# Check Porcupine wake word files
echo "Checking Porcupine wake word files..."
mkdir -p stt/resources

if [ -f "stt/resources/Hey-Dude_en_raspberry-pi_v3_0_0.ppn" ]; then
    echo "✓ Raspberry Pi wake word file found"
else
    echo "⚠️  Raspberry Pi wake word file not found"
    echo "   Expected: stt/resources/Hey-Dude_en_raspberry-pi_v3_0_0.ppn"
fi

echo ""
echo "🔧 Required Setup:"
echo "1. Get your AccessKey from https://console.picovoice.ai/"
echo "2. Set environment variables:"
echo "   export PORCUPINE_ACCESS_KEY='your_access_key_here'"
echo "   export WEBHOOK_URL='your_webhook_url_here'"
echo ""

# Configure audio settings for Raspberry Pi
echo "Configuring audio settings..."

# Create asound.conf for better audio handling
sudo tee /etc/asound.conf > /dev/null <<EOF
pcm.!default {
    type asym
    capture.pcm "mic"
    playback.pcm "speaker"
}

pcm.mic {
    type plug
    slave {
        pcm "hw:1,0"
    }
}

pcm.speaker {
    type plug
    slave {
        pcm "hw:0,0"
    }
}
EOF

echo ""
echo "🎯 Performance Optimizations for RPi 3:"
echo "1. The script uses 'tiny' Whisper model for faster processing"
echo "2. Speaker identification is optional (disabled by default for performance)"
echo "3. Simple energy-based VAD fallback when PyTorch VAD is not available"
echo "4. Larger audio chunks (1024) for better RPi performance"
echo ""

echo "📋 Next Steps:"
echo "1. Complete the manual Porcupine setup above"
echo "2. Test audio: arecord -d 5 test.wav && aplay test.wav"
echo "3. Set environment variables in .env file"
echo "4. Run: python3 stt/main.py"
echo ""

echo "✅ Raspberry Pi 3 setup complete!"
echo "Note: First run may be slower as models are downloaded and cached."