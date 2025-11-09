#!/usr/bin/env python3
"""
Raspberry Pi STT System Test Script

This script tests all components of the STT system to ensure
everything is working correctly on Raspberry Pi 3.
"""

import os
import sys
import platform
import numpy as np
import sounddevice as sd
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def test_platform():
    """Test platform detection."""
    print_header("PLATFORM DETECTION TEST")
    print(f"System: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print(f"Node: {platform.uname().node}")
    
    is_rpi = platform.machine().lower().startswith('arm') or 'raspi' in platform.uname().node.lower()
    print(f"Detected as Raspberry Pi: {is_rpi}")
    
    if is_rpi:
        print("✓ Platform detection working correctly")
        return True
    else:
        print("⚠️  Not detected as Raspberry Pi - optimizations may not apply")
        return True

def test_audio_devices():
    """Test audio device detection."""
    print_header("AUDIO DEVICES TEST")
    try:
        devices = sd.query_devices()
        print("Available audio devices:")
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append("INPUT")
            if device['max_output_channels'] > 0:
                device_type.append("OUTPUT")
            
            print(f"  {i}: {device['name']} ({', '.join(device_type)})")
        
        # Check for default devices
        try:
            default_input = sd.query_devices(kind='input')
            default_output = sd.query_devices(kind='output')
            print(f"\nDefault input: {default_input['name']}")
            print(f"Default output: {default_output['name']}")
            print("✓ Audio devices detected successfully")
            return True
        except Exception as e:
            print(f"❌ Error getting default devices: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error querying audio devices: {e}")
        return False

def test_microphone():
    """Test microphone recording."""
    print_header("MICROPHONE TEST")
    try:
        print("Testing microphone for 3 seconds...")
        print("Speak now!")
        
        # Record 3 seconds of audio
        duration = 3
        sample_rate = 16000
        audio_data = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='int16')
        sd.wait()  # Wait for recording to complete
        
        # Check if we got audio data
        if len(audio_data) > 0:
            # Calculate RMS to check for audio activity
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            print(f"RMS level: {rms:.6f}")
            
            if rms > 0.001:  # Threshold for audio activity
                print("✓ Microphone is capturing audio")
                return True
            else:
                print("⚠️  Microphone not detecting significant audio - check connections")
                return False
        else:
            print("❌ No audio data captured")
            return False
            
    except Exception as e:
        print(f"❌ Error testing microphone: {e}")
        return False

def test_dependencies():
    """Test required dependencies."""
    print_header("DEPENDENCIES TEST")
    
    required_modules = [
        'sounddevice',
        'numpy', 
        'requests',
        'pvporcupine',
        'faster_whisper'
    ]
    
    optional_modules = [
        'torch',
        'torchaudio', 
        'resemblyzer'
    ]
    
    all_good = True
    
    print("Required modules:")
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ❌ {module} - MISSING")
            all_good = False
    
    print("\nOptional modules:")
    for module in optional_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ⚠️  {module} - not installed (optional)")
    
    return all_good

def test_porcupine_setup():
    """Test Porcupine wake word setup."""
    print_header("PORCUPINE WAKE WORD TEST")
    
    # Check for access key
    access_key = os.getenv("PORCUPINE_ACCESS_KEY")
    if not access_key:
        print("❌ PORCUPINE_ACCESS_KEY environment variable not set")
        return False
    else:
        print(f"✓ Access key found: {access_key[:10]}...")
    
    # Check for keyword files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(script_dir, "stt", "resources")
    
    keyword_files = [
        "Hey-Dude_en_raspberry-pi_v3_0_0.ppn",
        "Hey-Dude_en_linux_v3_0_0.ppn", 
        "Hey-Dude_en_windows_v3_0_0.ppn"
    ]
    
    found_keyword = False
    for keyword_file in keyword_files:
        keyword_path = os.path.join(resources_dir, keyword_file)
        if os.path.exists(keyword_path):
            print(f"✓ Found keyword file: {keyword_file}")
            found_keyword = True
            break
    
    if not found_keyword:
        print("❌ No Porcupine keyword file found")
        print("   Expected files:")
        for kf in keyword_files:
            print(f"     - {kf}")
        return False
    
    # Try to initialize Porcupine
    try:
        import pvporcupine
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[keyword_path]
        )
        porcupine.delete()  # Clean up
        print("✓ Porcupine initialization successful")
        return True
    except Exception as e:
        print(f"❌ Porcupine initialization failed: {e}")
        return False

def test_whisper():
    """Test Whisper model loading."""
    print_header("WHISPER MODEL TEST")
    
    try:
        from faster_whisper import WhisperModel
        print("Loading Whisper model (this may take a moment)...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✓ Whisper model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        return False

def test_environment():
    """Test environment variables."""
    print_header("ENVIRONMENT VARIABLES TEST")
    
    required_vars = ["PORCUPINE_ACCESS_KEY"]
    optional_vars = ["WEBHOOK_URL"]
    
    all_required = True
    
    print("Required environment variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {value[:20]}...")
        else:
            print(f"  ❌ {var}: NOT SET")
            all_required = False
    
    print("\nOptional environment variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {value[:30]}...")
        else:
            print(f"  ⚠️  {var}: not set")
    
    return all_required

def main():
    """Run all tests."""
    print("🔧 Raspberry Pi STT System Test")
    print(f"Test started at: {datetime.now()}")
    
    tests = [
        ("Platform Detection", test_platform),
        ("Dependencies", test_dependencies),
        ("Environment Variables", test_environment),
        ("Audio Devices", test_audio_devices),
        ("Microphone", test_microphone),
        ("Porcupine Setup", test_porcupine_setup),
        ("Whisper Model", test_whisper)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Raspberry Pi STT system should work correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please address the issues above.")
        print("   Refer to README-RPi.md for troubleshooting guidance.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)