# Setup Passwordless Sudo for PowerManager

The PowerManager needs to execute `rocm-smi --setperflevel` commands without requiring a password prompt.

## Option 1: Sudoers Configuration (Recommended)

Add a sudoers rule to allow passwordless execution of rocm-smi for your user:

```bash
sudo visudo -f /etc/sudoers.d/rocm-smi
```

Add this line (replace `robiloo` with your username):
```
robiloo ALL=(ALL) NOPASSWD: /opt/rocm/bin/rocm-smi
```

Or if rocm-smi is in a different location, find it first:
```bash
which rocm-smi
```

Then use that full path in the sudoers file.

## Option 2: Add User to Video Group

ROCm commands might work without sudo if your user is in the `video` and `render` groups:

```bash
sudo usermod -a -G video,render $USER
```

Then log out and back in for the changes to take effect.

## Option 3: Set Permissions on rocm-smi (Less Secure)

```bash
sudo chmod u+s /opt/rocm/bin/rocm-smi
```

⚠️ This is less secure as it allows any user to run rocm-smi with elevated privileges.

## Verify Setup

Test that sudo works without password:
```bash
sudo rocm-smi --setperflevel auto
```

If it doesn't ask for a password, you're good to go!

## Alternative: Run PowerManager without Sudo

If you want to run without sudo, modify powerManager.py line 85 to remove "sudo" from the command:
```python
cmd = ["rocm-smi", "--setperflevel", level.value]
```

But this only works if you have the proper group permissions (Option 2 above).
