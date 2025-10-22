#!/bin/bash
# Setup script for AMD GPU sysfs write permissions
# This allows non-root users to control GPU clocks and fan speeds

set -e

echo "=========================================="
echo "AMD GPU Power Control Permission Setup"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Get the actual user (not root if using sudo)
ACTUAL_USER="${SUDO_USER:-$USER}"

echo "Step 1: Adding user '$ACTUAL_USER' to 'video' group..."
usermod -a -G video "$ACTUAL_USER"
echo "✓ User added to video group"
echo ""

echo "Step 2: Installing udev rules..."
cp 99-amdgpu-power-control.rules /etc/udev/rules.d/
chmod 644 /etc/udev/rules.d/99-amdgpu-power-control.rules
echo "✓ udev rules installed to /etc/udev/rules.d/"
echo ""

echo "Step 3: Reloading udev rules..."
udevadm control --reload-rules
echo "✓ udev rules reloaded"
echo ""

echo "Step 4: Triggering udev to apply rules..."
udevadm trigger
echo "✓ udev rules triggered"
echo ""

echo "Step 5: Verifying permissions..."
echo ""
echo "Checking sysfs file permissions:"
ls -l /sys/class/drm/card0/device/power_dpm_force_performance_level 2>/dev/null || echo "  ⚠ File not found"
ls -l /sys/class/drm/card0/device/pp_dpm_sclk 2>/dev/null || echo "  ⚠ File not found"

# Find hwmon device for card0
HWMON_PATH=$(find /sys/class/drm/card0/device/hwmon/hwmon* -maxdepth 0 -type d 2>/dev/null | head -1)
if [ -n "$HWMON_PATH" ]; then
    ls -l "$HWMON_PATH/pwm1" 2>/dev/null || echo "  ⚠ pwm1 not found"
else
    echo "  ⚠ hwmon device not found"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: You must LOG OUT and LOG BACK IN for group changes to take effect!"
echo ""
echo "After logging back in, verify permissions with:"
echo "  groups  # Should show 'video' in the list"
echo "  echo 'manual' > /sys/class/drm/card0/device/power_dpm_force_performance_level"
echo ""
echo "If you get 'Permission denied' after logging back in, you may need to reboot."
echo ""
