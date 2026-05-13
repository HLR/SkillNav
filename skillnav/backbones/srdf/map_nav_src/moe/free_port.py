import os
import socket

def is_port_in_use(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(('127.0.0.1', port))
    s.close()
    return result == 0

def kill_process_on_port(port):
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read()
                if b"uvicorn" in cmdline and str(port).encode() in cmdline:
                    print(f"Killing PID {pid} using port {port}")
                    os.kill(int(pid), 9)
                    return True
        except Exception:
            continue
    return False

PORT = 8001

if is_port_in_use(PORT):
    print(f"Port {PORT} is in use. Trying to free it...")
    if not kill_process_on_port(PORT):
        print("⚠️ Failed to kill process using /proc. Try rebooting container or change port.")
else:
    print(f"[✅] Port {PORT} is free.")