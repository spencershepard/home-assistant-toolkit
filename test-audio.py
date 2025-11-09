#!/usr/bin/env python3
"""
Audio Device Test Utility for Raspberry Pi STT Setup

This script helps you test your audio devices and determine if you need
custom ALSA configuration.
"""

import sounddevice as sd
import numpy as np
import time
import sys

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def test_default_devices():
    """Test if default audio devices work."""
    print_header("DEFAULT AUDIO DEVICES TEST")
    
    try:
        # Get default devices
        default_input = sd.query_devices(kind='input')
        default_output = sd.query_devices(kind='output')
        
        print(f"Default Input Device:")
        print(f"  Name: {default_input['name']}")
        print(f"  Channels: {default_input['max_input_channels']}")
        print(f"  Sample Rate: {default_input['default_samplerate']}")
        
        print(f"\nDefault Output Device:")
        print(f"  Name: {default_output['name']}")
        print(f"  Channels: {default_output['max_output_channels']}")
        print(f"  Sample Rate: {default_output['default_samplerate']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error querying default devices: {e}")
        return False

def list_all_devices():
    """List all available audio devices."""
    print_header("ALL AUDIO DEVICES")
    
    try:
        devices = sd.query_devices()
        
        print("Available audio devices:")
        for i, device in enumerate(devices):
            device_type = []
            if device['max_input_channels'] > 0:
                device_type.append(f"IN({device['max_input_channels']})")
            if device['max_output_channels'] > 0:
                device_type.append(f"OUT({device['max_output_channels']})")
            
            marker = " [DEFAULT]" if i == sd.default.device[0] or i == sd.default.device[1] else ""
            print(f"  {i:2d}: {device['name']} ({', '.join(device_type)}){marker}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing devices: {e}")
        return False

def test_microphone_recording():
    """Test microphone recording with default device."""
    print_header("MICROPHONE RECORDING TEST")
    
    try:
        print("Testing microphone recording (5 seconds)...")
        print("🎤 Speak now or make some noise!")
        
        # Record audio
        duration = 5
        sample_rate = 16000
        print(f"Recording for {duration} seconds...")
        
        audio_data = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
        sd.wait()
        
        # Analyze the recording
        rms = np.sqrt(np.mean(audio_data ** 2))
        max_amplitude = np.max(np.abs(audio_data))
        
        print(f"\nRecording Analysis:")
        print(f"  RMS Level: {rms:.6f}")
        print(f"  Max Amplitude: {max_amplitude:.6f}")
        print(f"  Non-zero samples: {np.count_nonzero(audio_data)}")
        
        if rms > 0.001:
            print("✅ Good audio signal detected!")
            
            # Offer to play back the recording
            try:
                response = input("\nPlay back the recording? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    print("🔊 Playing back recording...")
                    sd.play(audio_data, sample_rate)
                    sd.wait()
                    print("✅ Playback complete")
            except KeyboardInterrupt:
                print("\nPlayback cancelled")
            
            return True
        else:
            print("⚠️  Very low audio signal - check microphone connection")
            return False
            
    except Exception as e:
        print(f"❌ Error during microphone test: {e}")
        return False

def test_specific_device():
    """Test recording with a specific device."""
    print_header("SPECIFIC DEVICE TEST")
    
    try:
        # List devices first
        devices = sd.query_devices()
        input_devices = [(i, dev) for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
        
        if not input_devices:
            print("❌ No input devices found")
            return False
        
        print("Available input devices:")
        for i, (device_id, device) in enumerate(input_devices):
            print(f"  {i}: [{device_id}] {device['name']}")
        
        try:
            choice = input(f"\nSelect device to test (0-{len(input_devices)-1}, or Enter for default): ").strip()
            
            if choice == "":
                device_id = None
                print("Using default device")
            else:
                device_index = int(choice)
                device_id = input_devices[device_index][0]
                device_name = input_devices[device_index][1]['name']
                print(f"Testing device: {device_name}")
            
            # Test recording with selected device
            print("Recording 3 seconds...")
            audio_data = sd.rec(int(3 * 16000), 
                              samplerate=16000, 
                              channels=1, 
                              dtype='float32',
                              device=device_id)
            sd.wait()
            
            rms = np.sqrt(np.mean(audio_data ** 2))
            print(f"RMS Level: {rms:.6f}")
            
            if rms > 0.001:
                print("✅ Device working correctly!")
                return True
            else:
                print("⚠️  Low signal from this device")
                return False
                
        except (ValueError, IndexError):
            print("❌ Invalid device selection")
            return False
            
    except Exception as e:
        print(f"❌ Error testing specific device: {e}")
        return False

def show_recommendations():
    """Show recommendations based on system state."""
    print_header("RECOMMENDATIONS")
    
    print("Based on your audio setup:")
    print()
    print("✅ If default devices work fine:")
    print("   → No additional configuration needed")
    print("   → The STT system will work out of the box")
    print()
    print("⚠️  If you have multiple audio devices or conflicts:")
    print("   → Consider running the setup script with custom audio config")
    print("   → Or manually create /etc/asound.conf")
    print()
    print("❌ If no devices work:")
    print("   → Check USB connections")
    print("   → Run: lsusb | grep -i audio")
    print("   → Check if drivers are loaded: lsmod | grep snd")
    print()
    print("🔧 STT System Device Selection:")
    print("   → Uses DEVICE_INDEX = None (system default)")
    print("   → This works for 90% of setups")
    print("   → Only change if you have specific requirements")

def main():
    """Run the audio device test suite."""
    print("🔊 Raspberry Pi Audio Device Test Suite")
    print("This will help determine if you need custom audio configuration.")
    
    tests = [
        ("List All Devices", list_all_devices),
        ("Test Default Devices", test_default_devices),
        ("Test Microphone Recording", test_microphone_recording),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            results[test_name] = test_func()
            
            if not results[test_name]:
                response = input(f"\n{test_name} had issues. Continue anyway? (Y/n): ").strip().lower()
                if response in ['n', 'no']:
                    break
                    
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Optional specific device test
    try:
        response = input(f"\nTest a specific audio device? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            test_specific_device()
    except KeyboardInterrupt:
        pass
    
    # Show recommendations
    show_recommendations()
    
    # Summary
    print_header("SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed >= 2:  # If most tests passed
        print("\n🎉 Your audio setup looks good!")
        print("You probably DON'T need custom asound.conf configuration.")
        print("The STT system should work with default settings.")
    else:
        print("\n⚠️  Audio setup may need attention.")
        print("Consider custom configuration or troubleshooting.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)