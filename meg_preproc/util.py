import sys

def info(s, marker='='):
    n = len(s)
    stderr('\n\n' + marker * n + '\n' + s + '\n' + marker * n + '\n\n')

def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()
