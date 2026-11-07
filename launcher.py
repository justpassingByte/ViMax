import os
import sys
import time
import socket
import webbrowser
import subprocess

PORT = 4173
HOST = '127.0.0.1'

def is_port_open(host=HOST, port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.8)
        return s.connect_ex((host, port)) == 0

def main():
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))

    web_dir = os.path.join(app_dir, 'web')
    bundled_node = os.path.join(app_dir, 'bin', 'node.exe')
    bundled_python = os.path.join(app_dir, '.venv', 'Scripts', 'python.exe')

    node_cmd = bundled_node if os.path.exists(bundled_node) else 'node'
    
    env = os.environ.copy()
    if os.path.exists(bundled_python):
        env['VIMAX_PYTHON_CMD'] = bundled_python

    # If server isn't listening yet, start it in the background
    if not is_port_open(HOST, PORT):
        creation_flags = 0x08000000 if os.name == 'nt' else 0  # CREATE_NO_WINDOW
        subprocess.Popen(
            [node_cmd, 'server.mjs'],
            cwd=web_dir,
            env=env,
            creationflags=creation_flags
        )
        for _ in range(40):
            time.sleep(0.4)
            if is_port_open(HOST, PORT):
                break

    # Open default browser to the studio URL
    webbrowser.open(f'http://{HOST}:{PORT}')

if __name__ == '__main__':
    main()
